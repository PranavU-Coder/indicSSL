"""Masked Modeling Duo (M2D) & M2D-CLAP Runtime class/functions.
"""

import sys
sys.path.append('..')  # workaround for using heareval with `pip install -e .`

import logging
from pathlib import Path
import re
import os

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange
import nnAudio.features

from . import models_mae
from .timm_layers_pos_embed import resample_abs_pos_embed 
from transformers import AutoTokenizer, AutoModel, BertModel


class Config:
    weight_file = ''
    feature_d = 768 * 5

    model = ''
    input_size = [80, 208]
    patch_size = [16, 16]
    sr = '16k'
    training_mask = 0.0
    flat_features = False
    encoder_only = True  # For using in fine-tuning
    dur_frames = None    # None for no desired number of frames
    freeze_embed = None  # Set True if freezing PatchEmbed during fine-tuning [2211.09359] How to Fine-Tune Vision Models with SGD


def parse_sizes_by_name(name):
    # Parse parameters. "m2d_vit_base-80x1001p16x16p16kpXXXpYYY" -> input size: 80x1001, patch size: 16x16, sr: 16k, extra parameters: XXX and YYY
    model_cls = name.split('-')[0]
    params = name.split('-')[1]
    params = params.split('p')
    params, extra = params[:3], params[3:]
    input_str, patch_str, sr = params[0], params[1], params[2] if len(params) > 2 else '16k'
    input_size = [int(a) for a in input_str.split('x')]
    patch_size = [int(a) for a in patch_str.split('x')]
    return input_size, patch_size, sr, model_cls, extra


def drop_non_model_weights(model, checkpoint, filename, except_for=[]):  # except_for=['running_mean', 'running_var']
    def not_an_exception(k):
        for ex in except_for:
            if ex in k: return False
        return True
    model_keys = [n for n, p in model.named_parameters()]
    new_ckpt = {}
    dropped = []
    for k in checkpoint:
        if k not in model_keys and not_an_exception(k):
            dropped.append(k)
            continue
        new_ckpt[k] = checkpoint[k]
    n_org = len(checkpoint.keys())
    n_cur = len(new_ckpt.keys())
    print(f' using {n_cur} parameters, while dropped {n_org - n_cur} out of {n_org} parameters from {Path(filename).parent/Path(filename).name}'
          if n_org > n_cur else f' using {n_cur} parameters from {Path(filename).parent/Path(filename).name}')
    print(' (included audio_proj params:', [k for k in new_ckpt.keys() if 'audio_proj' in k][:5])
    print(' (included text_proj params:', [k for k in new_ckpt.keys() if 'text_proj' in k][:5])
    print(' (dropped:', dropped[:5], ')' if len(dropped) < 5 else '...)')
    return new_ckpt


def load_evar_head_parameters(checkpoint, head_norm, head):
    # Load the weights of the task head trained in the EVAR fine-tuning.
    if 'module.head.norm.running_mean' in checkpoint:
        head_norm.load_state_dict({to_k: checkpoint[k] for to_k, k in {
            'running_mean':'module.head.norm.running_mean', 'running_var':'module.head.norm.running_var'}.items()})
        head.load_state_dict({to_k: checkpoint[k] for to_k, k in {
            'weight':'module.head.mlp.mlp.0.weight', 'bias':'module.head.mlp.mlp.0.bias'}.items()})
    else:
        print(' Not an EVAR checkpoint for loading head weights.')


def reformat_evar_ckpt(checkpoint):
    # The checkpoints saved in a EVAR fine-tuning has a prefix of "module.ar.runtime.backbone", the following removes it.
    new_ckpt = {}
    for k in checkpoint:
        new_k = k.replace('module.ar.runtime.backbone.', '')  # replace
        new_ckpt[new_k] = checkpoint[k]
    return new_ckpt


def extract_weight(checkpoint, root_name):
    # If no key matches the root_name, return the checkpoint unchanged.
    if not any(k.startswith(root_name) for k in checkpoint.keys()):
        return checkpoint
    # Keep only the items starts with the root_name
    new_ckpt = {k[len(root_name):]: v for k, v in checkpoint.items() if k.startswith(root_name)}
    return new_ckpt


def add_semantic_audio_proj(sem_mode, embed_dim):
    sem_params = {
        1: {'sem_depth': 1, 'sem_heads': 1, 'sem_mlp_ratio': 1},
        2: {'sem_depth': 2, 'sem_heads': 1, 'sem_mlp_ratio': 1},
        3: {'sem_depth': 3, 'sem_heads': 1, 'sem_mlp_ratio': 2},
        4: {'sem_depth': 4, 'sem_heads': 1, 'sem_mlp_ratio': 2},
    }[sem_mode]
    audio_proj = models_mae.AudioToSemantic(embed_dim=embed_dim, **sem_params)
    return audio_proj


def make_it_CLAP_if_needed(model, checkpoint):
    # Return if already a CLAP model
    if hasattr(model, 'audio_proj') or checkpoint is None: return
    # Add projectors if needed
    if 'audio_proj.0.weight' in checkpoint.keys():
        proj_hidden_dim, embed_dim = checkpoint['audio_proj.0.weight'].shape
        off_emb_dim = checkpoint['audio_proj.2.weight'].shape[0]
        model.audio_proj = models_mae.get_MLP_projector(embed_dim, embed_dim, embed_dim)
    if 'audio_proj.sem_token' in checkpoint.keys():
        embed_dim = checkpoint['audio_proj.sem_token'].shape[-1]
        sem_blocks_nums = [int(k.split('.')[2]) for k in checkpoint.keys() if k.startswith('audio_proj.sem_blocks.')]
        sem_mode = max(sem_blocks_nums) + 1
        model.audio_proj = add_semantic_audio_proj(sem_mode, embed_dim)
    if 'text_proj.weight' in checkpoint.keys():
        dim = checkpoint['text_proj.weight'].shape
        model.text_proj = torch.nn.Linear(dim[1], dim[0])
    if 'text_proj.2.weight' in checkpoint.keys():
        dim = checkpoint['text_proj.2.weight'].shape
        model.text_proj = models_mae.get_MLP_projector(dim[1], dim[1], dim[0])
    ## For M2D-CLAP (2025) ablations
    if hasattr(model, 'text_proj') and not hasattr(model, 'audio_proj'):
        model.audio_proj = torch.nn.Identity()


def parse_clap_type(name):
    # Parse parameters. "m2d_clap_base-80x1001p16x16p16kpA" -> input size: 80x1001, patch size: 16x16, sr: 16k, extra: A
    params = str(name).split('-')[1]
    params = params.split('p')
    params, extra = params[:3], params[3:]
    if len(extra) == 0:
        return 'A'
    assert extra[0] in 'ABCDELMNQR'
    text_encoder_name = {'A': 'GTE base', 'B': 'BERT base', 'C': 'CLIP-L', 'E': 'BERT large', 'L': 'GTE large',
                         'M': 'Modern BERT Base', 'N': 'NV-Embed-v2', 'Q': 'gte-Qwen2-7B-instruct', 'R': 'RoBERTa base'}
    logging.info(f' using text encoder: {text_encoder_name[extra[0]]}')
    return extra[0]


def clap_off_emb_dim(param_extra):
    if len(param_extra) == 0:
        return 768
    return {'A': 768, 'B': 768, 'C': 1024, 'D': 1024, 'L': 1024, 'M': 768, 'N': 4096, 'Q': 3584, 'R': 768}[param_extra[0]]


def parse_clap_text_encoder_weight(param_extra, cfg, ckpt_cfg=None):
    # param_extra[0]=text encoder type, param_extra[1]=text encoder included (TI)
    if len(param_extra) <= 1:
        return None
    if len(param_extra) > 1 and param_extra[1] == 'TI':  # text encoder included
        return cfg.weight_file
    assert False, f'unknown extra parameters: {param_extra}'


def get_backbone(args, weight_file, encoder_only, dur_frames):
    # Find the model parameters
    try:
        args.input_size, args.patch_size, args.sr, args.model, extra = parse_sizes_by_name(Path(weight_file).parent.name)
    except:
        args.input_size, args.patch_size, args.sr, args.model, extra = parse_sizes_by_name(Path(weight_file).stem)
    if dur_frames is not None:
        org_input_size = args.input_size.copy()
        args.input_size[1] = dur_frames

    if encoder_only:
        args.model = args.model + '_encoder_only'
    if Path(weight_file).name.endswith('random'):
        checkpoint, ckpt_cfg = None, None
        dec_blocks_nums = [8 - 1] # fixed for random init -> 8 is the # of decoder blocks of M2D.
        norm_stats = [-7.1, 4.2]
        print(' **CAUTION: Random Weights**')
        logging.info(' **CAUTION: Random Weights**')
    else:
        checkpoint = torch.load(weight_file, map_location='cpu', weights_only=False)
        ckpt_cfg = checkpoint['args'] if 'args' in checkpoint else None
        checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
        if checkpoint is not None:
            checkpoint = extract_weight(checkpoint, 'backbone.')  # convert from RuntimeM2D weights
        checkpoint = reformat_evar_ckpt(checkpoint)
        # Find the number of decoder blocks: "decoder_blocks.1." or "decoder_blocks.layers.1" or nothing
        dec_blocks_nums = [int(k.split('.')[2]) for k in checkpoint.keys() if k.startswith('decoder_blocks.layers')]
        if not dec_blocks_nums:
            dec_blocks_nums = [int(k.split('.')[1]) for k in checkpoint.keys() if k.startswith('decoder_blocks.')]
        if not dec_blocks_nums:
            dec_blocks_nums = [8 - 1] # fixed for random init -> 8 is the # of decoder blocks of M2D.
        norm_stats = checkpoint['norm_stats'] if 'norm_stats' in checkpoint else [-7.1, 4.2]
    args.decoder_depth = max(dec_blocks_nums) + 1

    model_args = dict(img_size=args.input_size, patch_size=args.patch_size, decoder_depth=args.decoder_depth, norm_stats=norm_stats)
    if args.model.startswith('m2d_x_vit') or args.model.startswith('m2d_as_vit'):
        off_emb_dim = 3840 if (checkpoint is None) or ('offline_predictor.weight' not in checkpoint) else checkpoint['offline_predictor.weight'].shape[0]
        model_args['off_emb_dim'] = off_emb_dim
    if args.model.startswith('m2d_clap'):
        model_args['off_emb_dim'] = clap_off_emb_dim(extra)
    args.text_encoder_weight = parse_clap_text_encoder_weight(extra, args, ckpt_cfg)  # any model can have a text encoder

    # create model
    print(f'Creating model: {args.model}({model_args})')
    model = models_mae.__dict__[args.model](**model_args)
    make_it_CLAP_if_needed(model, checkpoint)

    # load weights
    if checkpoint:
        # interpolate pos_embed
        if dur_frames is not None:
            org_grid_size = [org_input_size[0] // args.patch_size[0], org_input_size[1] // args.patch_size[1]]
            new_grid_size = [args.input_size[0] // args.patch_size[0], args.input_size[1] // args.patch_size[1]]
            if org_grid_size[1] < new_grid_size[1]:
                checkpoint['pos_embed'] = resample_abs_pos_embed(checkpoint['pos_embed'], old_size=org_grid_size, new_size=new_grid_size)
                print(' resampled pos_embed from', org_grid_size, 'to', new_grid_size, '- new pos_embed shape is', checkpoint['pos_embed'].shape)
            elif org_grid_size[1] > new_grid_size[1]:
                posemb = checkpoint['pos_embed']
                _, _, D = posemb.shape
                posemb_prefix, posemb = posemb[:, :1], posemb[:, 1:]
                posemb = posemb.reshape(1, org_grid_size[0], org_grid_size[1], D)
                posemb = posemb[:, :, :new_grid_size[1], :].reshape(1, new_grid_size[0]*new_grid_size[1], D)
                checkpoint['pos_embed'] = torch.cat([posemb_prefix, posemb], dim=1)
                print(' trimmed pos_embed from', org_grid_size, 'to', new_grid_size, '- new pos_embed shape is', checkpoint['pos_embed'].shape)
        # backward compatibility: norm_stats
        checkpoint['norm_stats'] = checkpoint['norm_stats'] if 'norm_stats' in checkpoint else torch.tensor(norm_stats)

        # remove non-model parameters (i.e. for using encoder only model)
        dropped = drop_non_model_weights(model, checkpoint, weight_file)
        msg = model.load_state_dict(dropped)
        print(msg)
        logging.info(msg)

    # set normalization statistics
    args.mean, args.std = norm_stats
    print(f' using norm_stats: {args.mean}, {args.std}')

    model.eval()
    return model, checkpoint


def get_to_melspec(cfg):
    if cfg.sr == '16k':
        cfg.sample_rate, cfg.n_fft, cfg.window_size, cfg.hop_size = 16000, 400, 400, 160
        cfg.n_mels, cfg.f_min, cfg.f_max = 80, 50, 8000
    elif cfg.sr == '32k':
        cfg.sample_rate, cfg.n_fft, cfg.window_size, cfg.hop_size = 32000, 800, 800, 320
        cfg.n_mels, cfg.f_min, cfg.f_max = 80, 50, 16000
    else:
        assert False, f'Unknown input size: {cfg.input_size}'

    to_spec = nnAudio.features.MelSpectrogram(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.window_size,
        hop_length=cfg.hop_size,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        center=True,
        power=2,
        verbose=False,
    )
    logging.info(f'Runtime MelSpectrogram({cfg.sample_rate}, {cfg.n_fft}, {cfg.window_size}, {cfg.hop_size}, '
                 + f'{cfg.n_mels}, {cfg.f_min}, {cfg.f_max}):')
    logging.info(to_spec)
    return to_spec


def get_timestamps(cfg, batch_audio, x):  # Returns timestamps in milliseconds.
    audio_len = len(batch_audio[0])
    sec = audio_len / cfg.sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000 # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts


class RuntimeM2D(nn.Module):
    def __init__(self, cfg=Config(), weight_file=None, training_mask=0.0, encoder_only=None, dur_frames=None, num_classes=None, freeze_embed=None, flat_features=None, random_mask=False):
        super().__init__()
        cfg.weight_file = weight_file or cfg.weight_file
        cfg.training_mask = training_mask if training_mask > 0.0 else cfg.training_mask
        self.cfg = cfg
        cfg.encoder_only = cfg.encoder_only if encoder_only is None else encoder_only
        cfg.dur_frames = cfg.dur_frames if dur_frames is None else dur_frames
        cfg.freeze_embed = cfg.freeze_embed if freeze_embed is None else freeze_embed
        cfg.flat_features = cfg.flat_features if flat_features is None else flat_features

        # Create backbone model.
        self.backbone, checkpoint = get_backbone(cfg, cfg.weight_file, cfg.encoder_only, cfg.dur_frames)
        # Finalize feature dimension. (768 if flat_features else 768*5=3840)
        d = self.backbone.pos_embed.shape[-1]
        if self.is_training_mask() or \
         (num_classes is not None and 'module.head.mlp.mlp.0.weight' in checkpoint and checkpoint['module.head.mlp.mlp.0.weight'].shape[-1] == d):
            cfg.flat_features = True
        n_stack_feature = 1 if cfg.flat_features else (cfg.input_size[0] // cfg.patch_size[0])
        cfg.feature_d = d * n_stack_feature
        # Create head.
        if num_classes is not None:
            self.head_norm = torch.nn.BatchNorm1d(cfg.feature_d, affine=False)
            self.head = torch.nn.Linear(cfg.feature_d, num_classes)
            trunc_normal_(self.head.weight, std=2e-5)
            load_evar_head_parameters(checkpoint, self.head_norm, self.head)
        # Option: if training_mask is enabled, set structured masking
        if self.is_training_mask():
            if random_mask:
                self.backbone.set_random_unstructured_mask()
            else:
                self.backbone.set_random_structured_mask()
        # Option: freeze patch embedding ([2211.09359] How to Fine-Tune Vision Models with SGD)
        if cfg.freeze_embed:
            models_mae.set_requires_grad(self.backbone.patch_embed, False)
            logging.info(' ** Freeze patch_embed **')
            logging.info(self.backbone.patch_embed)

        logging.info(str(cfg))
        logging.info(f'Model input size: {cfg.input_size}')
        logging.info(f'Using weights: {cfg.weight_file}')
        logging.info(f'training_mask: {cfg.training_mask}')
        logging.info(f'flat_features: {cfg.flat_features}')

        self.to_spec = get_to_melspec(cfg)
        self.sample_rate = cfg.sample_rate
        self.eval()

    def is_training_mask(self):
        return self.cfg.training_mask > 0.0

    def to_feature(self, batch_audio):
        x = self.to_spec(batch_audio)
        x = (x + torch.finfo().eps).log()
        x = x.unsqueeze(1)
        return x

    def normalize_batch(self, x):
        x = (x - self.cfg.mean) / self.cfg.std
        return x

    def to_normalized_spec(self, batch_audio):
        x = self.to_feature(batch_audio)
        x = self.normalize_batch(x)
        return x

    def encode_lms(self, x, return_layers=False, average_per_time_frame=False):
        if self.cfg.dur_frames is not None:
            return self.encode_lms_w_duration(x, return_layers=return_layers)

        patch_fbins = self.backbone.grid_size()[0]
        unit_frames = self.cfg.input_size[1]
        patch_frames = self.backbone.patch_size()[1]
        embed_d = self.backbone.patch_embed.proj.out_channels
        chunks = (x.shape[-1] + unit_frames - 1) // unit_frames
        pad_frames = (patch_frames - (x.shape[-1] % unit_frames % patch_frames)) % patch_frames
        if pad_frames > 0:
            x = torch.nn.functional.pad(x, (0, pad_frames))

        embeddings = []
        if self.cfg.flat_features:
            # Flatten all patch embeddings
            mask_ratio = self.cfg.training_mask if self.training else 0.0
            self.caution_mask_ratio(mask_ratio)
            for i in range(chunks):
                emb, *_ = self.backbone.forward_encoder(x[..., i*unit_frames:(i+1)*unit_frames], mask_ratio=mask_ratio, return_layers=return_layers, adjust_short=True)
                cls_token, emb = emb[..., :1, :], emb[..., 1:, :]
                if average_per_time_frame:
                    emb = rearrange(emb, 'b (f t) d -> b t d f', f=patch_fbins, d=embed_d).mean(-1)
                embeddings.append(emb)
        else:
            # Stack embeddings along time frame
            for i in range(chunks):
                emb, *_ = self.backbone.forward_encoder(x[..., i*unit_frames:(i+1)*unit_frames], mask_ratio=0., return_layers=return_layers, adjust_short=True)
                cls_token, emb = emb[..., :1, :], emb[..., 1:, :]
                if len(emb.shape) > 3:
                    emb = rearrange(emb, 'L b (f t) d -> L b t (f d)', f=patch_fbins, d=embed_d)  # Layer-wise embeddings
                else:
                    emb = rearrange(emb, 'b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)
                embeddings.append(emb)
        # Concatenate chunks in the time axis
        x = torch.cat(embeddings, axis=-2)
        return x if len(x.shape) == 3 else [x_ for x_ in x]

    def caution_mask_ratio(self, mask_ratio):
        if hasattr(self, 'done_caution_mask_ratio'): return
        self.done_caution_mask_ratio = True
        if mask_ratio > 0.0:
            logging.info(f' *CAUTION* training_mask (mask_ratio): {mask_ratio} > 0.0')
            print(f' *CAUTION*  training_mask (mask_ratio): {mask_ratio} > 0.0')

    def encode_lms_w_duration(self, x, return_layers=False):
        # Encode x without splitting into chunks.
        mask_ratio = self.cfg.training_mask if self.training else 0.0
        x, *_ = self.backbone.forward_encoder(x, mask_ratio=mask_ratio, return_layers=return_layers, adjust_short=True)
        x = x[..., 1:, :]  # Remove cls_token
        if not self.cfg.flat_features:
            # Stack embeddings along time frame
            patch_fbins = self.backbone.grid_size()[0]
            embed_d = self.backbone.patch_embed.proj.out_channels
            if len(x.shape) > 3:
                x = rearrange(x, 'L b (f t) d -> L b t (f d)', f=patch_fbins, d=embed_d)  # Layer-wise embeddings
            else:
                x = rearrange(x, 'b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)

        return x if len(x.shape) == 3 else [x_ for x_ in x]

    def encode(self, batch_audio, average_per_time_frame=False):
        x = self.to_normalized_spec(batch_audio)
        return self.encode_lms(x, average_per_time_frame=average_per_time_frame)

    def forward(self, batch_audio, average_per_time_frame=False):
        x = self.encode(batch_audio, average_per_time_frame=average_per_time_frame)
        if hasattr(self, 'head'):
            x = x.mean(1)  # B, D
            x = self.head_norm(x.unsqueeze(-1)).squeeze(-1)
            x = self.head(x)
        return x

    def get_scene_embeddings(self, batch_audio):
        x = self.encode(batch_audio)
        x = torch.mean(x, dim=1)
        return x

    def get_timestamp_embeddings(self, batch_audio):
        x = self.encode(batch_audio, average_per_time_frame=True)
        ts = get_timestamps(self.cfg, batch_audio, x)
        return x, ts

    def forward_frames(self, batch_audio):
        x, ts = self.get_timestamp_embeddings(batch_audio)
        if hasattr(self, 'head'):
            x = self.head_norm(x.transpose(-1, -2)).transpose(-2, -1)
            x = self.head(x)
        return x, ts

    def reconstruct(self, lms, mask_ratio, start_frame=0):
        """(Not for M2D, MAE only) A helper function to get reconstruction results.
        Use `lms_to_wav` if you may also want to convert the reconstruction results to wavs.
        **Note** this does *not* process the entire LMS frames but rather crops them from the start_frame with the duration of the model's unit frame.
        """
        # trim frames
        unit_frames = self.backbone.patch_embed.img_size[1]
        last_frame = start_frame + unit_frames
        lms_cropped = lms[..., start_frame:last_frame]
        # raw reconstruction
        with torch.no_grad():
            loss, recons, errormap, mask = self.backbone.forward_viz(lms_cropped, mask_ratio)

        return loss, lms_cropped, recons, errormap, mask

    def project_audio(self, audio_embeddings):
        if not hasattr(self.backbone.audio_proj, 'dont_average'):
            audio_embeddings = audio_embeddings.mean(dim=-2)
        audio_embeddings = self.backbone.audio_proj(audio_embeddings)
        return audio_embeddings

    def encode_clap_audio(self, batch_audio, is_lms=False):
        if is_lms:
            lms = self.normalize_batch(batch_audio)  # normalize LMS input
            audio_embeddings = self.encode_lms(lms)
        else:
            audio_embeddings = self.forward(batch_audio)
        audio_embeddings = self.project_audio(audio_embeddings)
        return audio_embeddings

    def encode_clap_text(self, batch_text, truncate=False):
        if not hasattr(self, 'text_encoder'):
            self.get_clap_text_encoder()
        text_embeddings = self.text_encoder(batch_text, truncate=truncate)
        if hasattr(self.backbone, 'text_proj'):
            text_embeddings = self.backbone.text_proj(text_embeddings)
        return text_embeddings

    def get_clap_text_encoder(self, weight_name=None, text_encoder_weight=None, remove_text_proj=False):
        text_encoder_weight = text_encoder_weight if text_encoder_weight is not None else \
            (self.cfg.text_encoder_weight if hasattr(self.cfg, 'text_encoder_weight') else None)
        weight_name = weight_name or self.cfg.weight_file
        self.text_encoder = get_text_encoder(weight_name, text_encoder_weight=text_encoder_weight)
        self.text_encoder = self.text_encoder.to(next(self.backbone.parameters()).device)
        if remove_text_proj and hasattr(self.backbone, 'text_proj'):
            d = None if hasattr(self.backbone.text_proj, 'weight') else self.backbone.text_proj[2].weight.shape[0]  # should be an MLP text projector
            del self.backbone.text_proj
            if d is not None:
                self.backbone.text_proj = models_mae.get_MLP_projector(d, d, d)
    

# For the CLAP models

class GTETextEncoder(nn.Module):
    def __init__(self, clip_weight="thenlper/gte-base"):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # To suppress warnings.

        self.tokenizer = AutoTokenizer.from_pretrained(clip_weight)
        self.model = AutoModel.from_pretrained(clip_weight)

    def __call__(self, texts, truncate=True, max_length=512):
        def average_pool(last_hidden_states, attention_mask):
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        device = next(self.model.parameters()).device
        batch_dict = self.tokenizer(texts, max_length=max_length, padding=True, truncation=truncate, return_tensors='pt')
        batch_dict['input_ids'] = batch_dict['input_ids'].to(device)
        batch_dict['token_type_ids'] = batch_dict['token_type_ids'].to(device)
        batch_dict['attention_mask'] = batch_dict['attention_mask'].to(device)
        outputs = self.model.to(device)(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings


class GTEL15Encoder(nn.Module):
    def __init__(self, clip_weight="Alibaba-NLP/gte-large-en-v1.5"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # To suppress warnings.

        self.model = SentenceTransformer(clip_weight, trust_remote_code=True)

    def __call__(self, texts, **kwargs):
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_tensor=True)
        return embeddings


class NVEmbedV2Encoder(nn.Module):
    def __init__(self, clip_weight="nvidia/NV-Embed-v2"):
        # https://huggingface.co/spaces/mteb/leaderboard https://huggingface.co/nvidia/NV-Embed-v2
        # https://arxiv.org/pdf/2405.17428
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # To suppress warnings.

        self.model = AutoModel.from_pretrained(clip_weight, trust_remote_code=True)

    def __call__(self, texts, **kwargs):
        embeddings = self.model.encode(texts, instruction="", max_length=32768)
        return embeddings


class CLIPLTextEncoder(nn.Module):
    def __init__(self, clip_weight="ViT-L/14"):
        super().__init__()
        import clip

        device = 'cpu'  # "cuda" if torch.cuda.is_available() else "cpu"
        self.clip = clip
        self.model, _ = clip.load(clip_weight, device=device)

    def __call__(self, texts, truncate=True, max_length=77):
        device = next(self.model.parameters()).device
        tokens = self.clip.tokenize(texts, context_length=max_length, truncate=True).to(device)
        embeddings = self.model.to(device).encode_text(tokens)
        return embeddings


class BertXEncoder(nn.Module):
    def __init__(self, clip_weight="google-bert/bert-base-uncased"):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # To suppress warnings.

        self.tokenizer = AutoTokenizer.from_pretrained(clip_weight)
        self.text_encoder = AutoModel.from_pretrained(clip_weight)

    def forward(self, batch_text, truncate=True, max_length=512):
        device = next(self.text_encoder.parameters()).device
        text_input = self.tokenizer(batch_text,
            padding='longest',
            truncation=truncate,
            max_length=max_length,
            return_tensors="pt").to(device)
        text_feats = self.text_encoder(input_ids=text_input.input_ids,
            attention_mask=text_input.attention_mask)[0]
        text_feats = text_feats[:, 0, :]
        return text_feats


def get_text_encoder(weight, text_encoder_weight=None):
    try:
        clap_type = parse_clap_type(Path(weight).parent.name)
    except:
        clap_type = parse_clap_type(Path(weight).stem)

    if clap_type == 'A':
        text_model = GTETextEncoder()
    if clap_type == 'B':
        text_model = BertXEncoder()
    if clap_type == 'C':
        text_model = CLIPLTextEncoder()
    if clap_type == 'E':
        text_model = BertXEncoder(clip_weight="google-bert/bert-large-uncased")
    if clap_type == 'L':
        text_model = GTEL15Encoder()
    if clap_type == 'M':
        text_model = BertXEncoder(clip_weight="answerdotai/ModernBERT-base")
    if clap_type == 'N':
        text_model = NVEmbedV2Encoder()
    if clap_type == 'Q':
        text_model = GTEL15Encoder(clip_weight="Alibaba-NLP/gte-Qwen2-7B-instruct")
    if clap_type == 'R':
        text_model = BertXEncoder(clip_weight="FacebookAI/roberta-base")

    if text_encoder_weight is not None:
        weights = torch.load(text_encoder_weight, map_location='cpu', weights_only=False)
        weights = weights['model'] if 'model' in weights else weights
        if ['module.ar.runtime.' in k for k in weights]:
            print(f' loading EVAR weight {text_encoder_weight} by removing "module.ar.runtime." from keys.')
            renamed = {k.replace('module.ar.runtime.', ''): weights[k] for k in weights}
            weights = renamed
        weights = extract_weight(weights, 'text_encoder.')
        print(f' using model.text_encoder from {text_encoder_weight}')
        text_model.load_state_dict(weights)
    return text_model
