"""M2D-CLAP Pre-training Script for Stages 2 and 2.1

- M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP
https://ieeexplore.ieee.org/document/11168481

- M2D-CLAP: Masked Modeling Duo Meets CLAP for Learning General-purpose Audio-Language Representation
https://www.isca-archive.org/interspeech_2024/niizumi24_interspeech.html
"""

import argparse
import datetime
import json
import numpy as np
import os
import time
import math
from pathlib import Path
import subprocess
import sys
import random
from typing import Iterable

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import matplotlib.pyplot as plt

import timm.optim
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from m2d import models_mae

from m2d.engine_pretrain_m2d import train_one_epoch_m2dx
from audio_dataset import SpectrogramDataset, get_files_no_sort, log_mixup_exp, pd
import common
from m2d.runtime_audio import RuntimeM2D, add_semantic_audio_proj, parse_clap_type


class CaptionSpecDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, files_main, files_bg_noise, file_caption, audio_frames, noise_ratio=0.0,
                 random_crop=True, n_norm_calc=10000) -> None:
        super().__init__()

        self.ds1 = SpectrogramDataset(folder=base_folder, files=files_main, crop_frames=audio_frames,
                random_crop=random_crop, norm_stats=[0.0, 1.0], n_norm_calc=0)
        print('ds1 norm_stats:', self.ds1.norm_stats)

        if noise_ratio > 0.0:
            self.ds2 = SpectrogramDataset(folder=base_folder, files=files_bg_noise, crop_frames=audio_frames,
                    random_crop=random_crop, norm_stats=[0.0, 1.0], n_norm_calc=0, repeat_short=True)
            print('ds2 norm_stats:', self.ds2.norm_stats)
        # for BG noise
        self.noise_ratio = noise_ratio
        self.bg_index = []
        # load captions
        self.caps = []
        for capfile in file_caption.split(','):
            caps = pd.read_csv(capfile).set_index('ytid')
            self.caps.append(caps)
            print('Caption', Path(capfile).stem, '-> captions:', len(caps), ' expample keys:', list(caps.index)[:5])
        self.file_caption = file_caption
        print('Using', len(self.caps), 'caption datasets.')

    def __len__(self):
        return len(self.ds1)

    def get_random_caption(self, ytid, folder_stem):
        cap_list = []
        # by the YouTube ID
        for cap in self.caps:
            if ytid in cap.index:
                cur = cap.loc[ytid].to_list()
                cap_list.extend(cur)
        # by the stem of the filename
        if len(cap_list) == 0:
            for cap in self.caps:
                if folder_stem in cap.index:
                    cur = cap.loc[folder_stem].to_list()
                    cap_list.extend(cur)
        if len(cap_list) == 0:
            print('MISSING', self.caps, ytid, folder_stem)
        return random.choice(cap_list)

    def __getitem__(self, index, fixed_noise=False):
        # load index sample
        clean = self.ds1[index]
        if self.noise_ratio > 0.0:
            # load random noise sample ### , while making noise floor zero
            noise = self.ds2[index if fixed_noise else self.get_next_bgidx()]
            # mix
            mixed = log_mixup_exp(noise, clean, self.noise_ratio) if self.noise_ratio < 1.0 else noise
        else:
            mixed = clean

        # load sample's caption
        filepath = self.ds1.df.file_name.values[index].split('/')
        ytid = filepath[-1][:11]
        stem = Path(filepath[-1]).stem
        folder = {'AudioSet_SL_flac': 'AudioSet_SL', 'BBC_Sound_Effects_flac': 'BBC_Sound_Effects',
            'FreeSound_flac': 'FreeSound', 'SoundBible_flac': 'SoundBible',
        }[filepath[-2]] if filepath[0] in ['wavcaps_lms', 'wavcaps_32klms'] else ''  # folder name conversion for wavcaps_lms
        if filepath[0] in ['clotho_lms', 'clotho_32klms']:
            stem = stem.replace(':', '_').replace('?', '_')
        caption = self.get_random_caption(ytid, folder + stem)

        return mixed, caption

    def get_next_bgidx(self):
        if len(self.bg_index) == 0:
            self.bg_index = torch.randperm(len(self.ds2)).tolist()
            # print(f'Refreshed the bg index list with {len(self.bg_index)} items: {self.bg_index[:5]}...')
        return self.bg_index.pop(0)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(crop_frames={self.ds1.crop_frames}, '
        format_string += f'folder_main={self.ds1.df.file_name.values[0].split("/")[0]}, '
        if self.noise_ratio > 0.: format_string += f'folder_bg={self.ds2.df.file_name.values[0].split("/")[0]}, '
        format_string += f'caption={self.file_caption})'
        return format_string


def build_captioned_dataset(cfg):
    # create dataset of (spectrogram, caption) pairs
    files_main = get_files_no_sort(cfg.csv_main)
    files_bg = get_files_no_sort(cfg.csv_bg_noise) if cfg.noise_ratio > 0. else []
    ds = CaptionSpecDataset(
        base_folder=cfg.data_path, files_main=files_main,
        files_bg_noise=files_bg,
        file_caption=cfg.file_caption,
        audio_frames=cfg.audio_frames,
        noise_ratio=cfg.noise_ratio,
        random_crop=True)

    return ds


def get_args_parser():
    """
    Usage: torchrun --nproc_per_node=4 -m clap.clap_only (arguments)

    Explicitly specified arguments for typical stage 2 pre-training:
    --epochs 30
    --eval_after 30
    --save_freq 30
    --data_path /path/to/my_data_lms
    --csv_main data/files_A_S_V_S_W_C_A_C_X_t_E_U.csv
    --mask_ratio 0.0  # 0.0 for stage 2.1, 0.3 for stage 2
    --base_model m2d_clap_vit_base-80x1001p16x16p16kpN-2025-stage1.1/weights_ep66it3124-0.49058_loss0.0147.pth
    --text_encoder B
    --finetune m2d_clap_vit_base-80x608p16x16p16kpBpTI-2025-stage2/checkpoint-30.pth  # for stage 2.1 only
    --seed 42

    Other implicit but important arguments:
    --audio_frames, default=1001
    --sem_mode 1
    """
    parser = argparse.ArgumentParser('CLAP only pre-training for M2D-CLAP', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--eval_after', default=10, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--feature_eval_freq', default=10, type=int, help='Feature label-free evaluation frequency.')
    parser.add_argument('--stop_at', default=-1, type=int)

    # Model parameters
    parser.add_argument('--base_model', default='m2d_clap_vit_base-80x608p16x16-240128/checkpoint-300.pth',
        type=str, metavar='MODEL', help='Base M2D-CLAP model.')

    parser.add_argument('--audio_frames', default=1001, type=int, help='Audio duration: 0=the input dur of the model, 1001=10s, etc.')

    parser.add_argument('--patch_size', default='16x16', type=str, help='patch size')
    parser.add_argument('--sr', default='16k', type=str, help='Sampling rate of the input audio.')  # just for adding to the output folder name
    parser.add_argument('--text_encoder', default='', type=str, metavar='TE', help='Text encoder type: A=GTEbase, B=BertBase, N=NVEmbedV2')
    parser.add_argument('--sem_mode', default=1, type=int, help='<Re-audio-projector> Replace audio projector with CLAP semantic mode (audio projector blocks).')
    parser.add_argument('--train_audio', action='store_true', help='Train audio encoder.')   

    parser.add_argument('--mask_ratio', default=0.3, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--random_mask', action='store_true')
    parser.add_argument('--target_layers', default='', type=str,
                        help='Experimental: layers to calculate target representations.')
    parser.add_argument('--bf16', action='store_true')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar="NORM",
                        help="Clip gradient norm (default: None, no clipping)")
    parser.add_argument('--optim', default='adamw', type=str, help='Optimizer adam or sdg')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=3e-6, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='data', type=str,
                        help='dataset path')
    parser.add_argument('--csv_main', default='data/files_A_S_V_S_W_C_A_C_X_t_E_U.csv', type=str, help='A CSV file to list sample files in the main dataset')

    parser.add_argument('--csv_bg_noise', default='', type=str,
                        help='A CSV file to list sample files in the BG noise dataset')
    parser.add_argument('--file_caption', default='data/rawcap_auto_acd.csv,data/rawcap_auto_acd_vggsound.csv,data/rawcap_wav_caps.csv,data/rawcap_audio_caps.csv,data/rawcap_clotho.csv',
                        type=str, help='CSV files for captions')
    parser.add_argument('--noise_ratio', default=0., type=float,
                        help='Noise mixing ratio')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--finetune', default='', help='Fine-tune checkpoint')
    parser.add_argument('--load_textenc', default='', help='Load text encoder weight')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--force_start_epoch', default=0, type=int, metavar='N',  # 0=always reset start epoch to 0 even if using --resume
                        help='start epoch for resuming')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank' if torch.__version__ >= "2.0.0" else '--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def get_optim(args, param_groups):
    if args.optim == 'adamw':
        return torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    elif args.optim == 'lars':
        return timm.optim.Lars(param_groups, lr=args.lr, momentum=0.9, weight_decay=0)
    elif args.optim == 'sgd':
        return torch.optim.SGD(param_groups, args.lr, momentum=0.9, weight_decay=0)
    assert False, f'Unsupported optimizer {args.optim}'


def set_layers_trainable(layer, trainable=False):
    for n, p in layer.named_parameters():
        p.requires_grad = trainable


def show_layers_trainable(layer, name='', show_all_trainable=True):
    total_params = sum(p.numel() for p in layer.parameters())
    total_trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f'{name}Total number of parameters: {total_params:,} (trainable {total_trainable_params:,})')
    trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
    frozen = [n for n, p in layer.named_parameters() if not p.requires_grad]
    print('Trainable parameters:', trainable if show_all_trainable else f'{trainable[:10]} ...')
    print('Others are frozen such as:', frozen[:3], '...' if len(frozen) >= 3 else '')


import util.lr_sched as lr_sched

def train_one_epoch_clap(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, loss_fn,
                    log_writer=None,
                    do_analysis: bool=False,
                    autocast_args: dict={},
                    args=None):
    """M2D-CLAP training loop."""
    _model = model if hasattr(model, 'logit_scale') else model.module
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x_online, x_target = samples[0].to(device, non_blocking=True), samples[1]

        with torch.cuda.amp.autocast(**autocast_args):
            audio_embs = _model.encode_clap_audio(x_online, is_lms=True)
            text_embs  = _model.encode_clap_text(x_target).to(device)
            loss = loss_fn(image_features=audio_embs, text_features=text_embs, logit_scale=_model.logit_scale.exp())
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=args.clip_grad, update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        # Clip logit scale if a CLAP model own it
        # Thanks to https://github.com/openai/CLIP/issues/46#issuecomment-793434211
        # logit scaling set as max 100
        _model.logit_scale.data = torch.clamp(_model.logit_scale.data, 0, 4.6052)  # 4.6052 == log(100)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # for CLIP model: logit_scale
    _logit_scale = getattr(_model, "logit_scale", None)
    if _logit_scale is not None:
        metric_logger.update(logit_scale=float(_logit_scale.detach().cpu()))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # basic matters
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    common.PrintLogger(f'{args.log_dir}/console.txt')

    # build model first
    model = RuntimeM2D(weight_file=args.base_model, training_mask=args.mask_ratio, flat_features=True, random_mask=args.random_mask)
    model.get_clap_text_encoder()
    model.logit_scale = torch.nn.Parameter(torch.ones([]) * 4.6052)  # 4.6052 == log(100)
    # replace audio projector
    if args.sem_mode > 0:
        print(' ** using semantic audio projector with', args.sem_mode, 'layers **')
        model.backbone.audio_proj = add_semantic_audio_proj(args.sem_mode, 768)
    # replace text encoder
    if re_text_encoder(model, Path(args.base_model).parent, Path(args.output_dir).stem):
        print(' ** replaced text encoder with', model.text_encoder)

    # set audio spec
    args.input_size = model.cfg.input_size
    if args.audio_frames > 0:
        print(f' using {args.audio_frames} audio frames from each data samples')
    args.audio_frames = args.audio_frames if args.audio_frames > 0 else model.cfg.input_size[1]

    # build dataset
    dataset_train = build_captioned_dataset(args)
    print(dataset_train)

    org_args_lr = args.lr
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / args.eff_batch_size) if org_args_lr is None else 'base lr: not effective')
    print("actual lr: %.2e" % args.lr)

    print("effective batch size: %d" % args.eff_batch_size)

    # build data loader
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # prepare the model
    model.to(device)
    print(model.cfg)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # freeze audio encoder
    set_layers_trainable(model.backbone, trainable=args.train_audio)
    set_layers_trainable(model.backbone.audio_proj, trainable=True)
    show_layers_trainable(model.backbone, name='<<Audio encoder>> ', show_all_trainable=False)
    show_layers_trainable(model.backbone.audio_proj, name='<<audio proj>> ', show_all_trainable=False)
    show_layers_trainable(model.text_encoder, name='<<Text encoder>> ', show_all_trainable=False)

    # set model as distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    try:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    except:
        print(' (for compatibility with timm) Switched add_weight_decay() to param_groups_weight_decay()')
        param_groups = timm.optim.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = get_optim(args, param_groups)
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print(f'Fine-tunning the checkpoint {args.finetune}')

    if args.load_textenc:
        checkpoint = torch.load(args.load_textenc, map_location='cpu')
        model_without_ddp.text_encoder.load_state_dict(checkpoint['model'], strict=True)
        print(f'Loaded text encoder weight {args.load_textenc}')

    if args.force_start_epoch >= 0:
        args.start_epoch = args.force_start_epoch

    print("{}".format(args).replace(', ', ',\n'))
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    last_subprocess = None

    # CLIP loss function
    loss_fn = models_mae.ClipLoss(rank=global_rank, world_size=num_tasks)

    for epoch in range(args.start_epoch, args.epochs):
        epoch1 = epoch + 1
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch_clap(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, loss_fn,
            log_writer=log_writer,
            do_analysis=(epoch1 % args.feature_eval_freq == 0),
            autocast_args=dict(dtype=torch.bfloat16) if args.bf16 else {},
            args=args
        )

        if args.output_dir and (epoch1 % args.save_freq == 0 or epoch1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch1)
            # run the external evaluator
            if args.eval_after <= epoch1 and epoch1 < args.epochs and misc.is_main_process():
                abspath = Path(f'{args.output_dir}/checkpoint-{epoch1}.pth').absolute()
                print('quick_eval', abspath)
                last_subprocess = subprocess.Popen(['/bin/bash', './quick_eval.sh', abspath])

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if args.stop_at > 0 and epoch1 >= args.stop_at:
            if last_subprocess is not None:
                last_subprocess.wait()
            print(f'Stop training by reaching args.stop_at epoch: {args.stop_at}')
            exit(0)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    del model_without_ddp, model, data_loader_train, optimizer, loss_scaler
    if misc.is_main_process():
        abspath = Path(f'{args.output_dir}/checkpoint-{epoch1}.pth').absolute()
        subprocess.call(['/bin/bash', 'clap/clap_eval.sh', abspath])
    return epoch1


arg_conf_defaults = {
    'csv_main': ('data/files_A_S_V_S_W_C_A_C_X_t_E_U.csv', 'M', 'path'),
    'mask_ratio': (0.3, 'mr', 'z'),
    'seed': (0, 's', 'asis'),
    'optim': ('adamw', 'O', 'asis'),
    'blr': (3e-6, 'blr', 'z'),
    'eff_batch_size': (2048, 'bs', 'asis'),
    'load_textenc': ('', 'TE', 'path'),
    'audio_frames': (1001, 'Dur', 'asis'),
    'random_mask': (False, 'R', 'b'),
    'train_audio': (False, 'trnA', 'b'),
    'sem_mode': (1, 'S', 'asis'),
    'finetune': ('', 'FT', 'b'),
}


def replace_text_encoder_spec(spec, new_text_encoder):
    if new_text_encoder != '':
        specs = spec.split('p')
        cur_text_encoder = specs[3]
        if new_text_encoder != cur_text_encoder:
            return 'p'.join(specs[:3] + [new_text_encoder] + specs[4:])
    return spec


def re_text_encoder(model, cur_name, new_name):
    cur_te = parse_clap_type(cur_name)
    new_te = parse_clap_type(new_name)
    if cur_te == new_te:
        return False
    model.get_clap_text_encoder(weight_name=new_name, remove_text_proj=True)
    return True


def complete_args():
    args = get_args_parser()
    args = args.parse_args()

    misc.init_distributed_mode(args)

    args.eff_batch_size = args.batch_size * misc.get_world_size()

    # m2d_clap_vit_base-80x608p16x16 -> m2d_clap_vit_base-80x608p16x16p16kpApTI
    parent = Path(args.base_model).parent.name.split('-')
    # For backward compatibility
    parent[1] = parent[1] if len(parent[1].split('p')) > 3 else (
        parent[1] + 'pA' if len(parent[1].split('p')) == 3 else parent[1] + 'p16kpA')
    # Replace text encoder model if specified
    parent[1] = replace_text_encoder_spec(parent[1], args.text_encoder)
    args.output_dir = f'{parent[0]}-{parent[1]}pTI-{"-".join(parent[2:])}'
    args.output_dir += f'-{common.arg_conf_str(args, defaults=arg_conf_defaults)}-{common.get_timestamp()[:6]}'

    if not args.log_dir:
        args.log_dir = args.output_dir
    return args


if __name__ == '__main__':
    args = complete_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
