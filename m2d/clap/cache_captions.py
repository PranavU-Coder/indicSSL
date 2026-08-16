"""Cache caption embeddings using a specified text encoder model.

Key Details:
    - Input: CSV files in 'data/' folder (must contain 'ytid' and caption columns).
    - Output: {cache_base}/{model_parent_folder}/capembs_{dataset}.npy
    - Optimization: Saves as float16 to conserve disk space.

Example Commands:
    # M2D-CLAP_2025 Standard
    $ python -m clap.cache_captions --bs 8
     --> Creates caption embeddings in `data/cache/m2d_clap_vit_base-80x608p16x16p16kpN-_N_V_E_m_b_2/capembs_*.npy`

    # M2D-CLAP_2024 (Legacy)
    $ python -m clap.cache_captions --model m2d_clap_vit_base-80x608p16x16p16kpA-_G_T_E/random
     --> Creates caption embeddings in `data/cache/m2d_clap_vit_base-80x608p16x16p16kpA-_G_T_E/capembs_*.npy`

    # Selected Datasets (C:Clotho, A:AudioCaps, S:Sound-VECaps)
    $ python -m clap.cache_captions --datasets CAS --bs 32
     --> Creates caption embeddings. ex) Sound-VECaps: `data/cache/m2d_clap_vit_base-80x608p16x16p16kpN-_N_V_E_m_b_2/capembs_sound_ve_caps.npy`

    # For ablation study
    $ python -m clap.cache_captions --model m2d_clap_vit_base-80x608p16x16p16kpQ-_G_T_E_Qwen2/random --bs 16
     --> Create caption embeddings in `data/cache/m2d_clap_vit_base-80x608p16x16p16kpQ-_G_T_E_Qwen2/capembs_*.npy`
"""
import argparse
import numpy as np
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch

from m2d.runtime_audio import RuntimeM2D


def get_args_parser():
    parser = argparse.ArgumentParser('Cache caption embeddings', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='m2d_clap_vit_base-80x608p16x16p16kpN-_N_V_E_m_b_2/random', type=str, help='Model name to specify the text encoder.')
    parser.add_argument('--cache_base', default='data/cache', type=str, help='Base cache folder name.')
    parser.add_argument('--out', default='', type=str, help='Output cache folder name if specified.')
    parser.add_argument('--bs', default=256, type=int, help='Batch size.')
    parser.add_argument('--datasets', default='CADVW', type=str, help='Datasets to convert.')  # CA4DVWS for all datasets, or any combination of C, A, 4, D, V, W, S

    return parser


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def convert(model, caption_file, out_file, bs):
    df = pd.read_csv(caption_file).set_index('ytid')
    if len(df.columns) == 1:
        cap_chunks = [c for c in chunks(df.values[:, 0].tolist(), bs)]
    else:
        cap_chunks = df.values.tolist()

    print(f'Encoding {caption_file}...')
    emb_chunks = []
    for i, caps in enumerate(tqdm(cap_chunks, mininterval=10.0)):
        with torch.no_grad():
            embeddings = model.encode_clap_text(caps).detach().cpu()
        emb_chunks.append(embeddings)

    if len(df.columns) == 1:
        embs = torch.cat(emb_chunks, dim=0).numpy().astype(np.float16)
    else:
        embs = torch.stack(emb_chunks).numpy().astype(np.float16)

    embdic = {y: c for y, c in zip(df.index, embs)}
    np.save(out_file, embdic)

    print(out_file, embs.shape, embs[:5])


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    # out folder name
    if not args.out:
        args.out = Path(args.cache_base)/Path(args.model).parent.name
    args.out.mkdir(parents=True, exist_ok=True)

    # build model first
    model = RuntimeM2D(weight_file=args.model)
    model.get_clap_text_encoder()
    model.backbone.text_proj = torch.nn.Identity()  # disable text projector
    model = model.to('cuda:0')

    print('Text encoder =', model.text_encoder)
    print('Text projector should be Identity =', model.backbone.text_proj, flush=True)
    print('Output folder =', args.out, flush=True)

    if 'C' in args.datasets:
        convert(model, 'data/rawcap_clotho.csv', args.out/'capembs_clotho.npy', args.bs)
    if 'A' in args.datasets:
        convert(model, 'data/rawcap_audio_caps.csv', args.out/'capembs_audio_caps.npy', args.bs)
    if '4' in args.datasets:
        convert(model, 'data/rawcap_ac_alt_4.csv', args.out/'capembs_ac_alt_4.npy', args.bs)
    if 'D' in args.datasets:
        convert(model, 'data/rawcap_auto_acd.csv', args.out/'capembs_auto_acd.npy', args.bs)
    if 'V' in args.datasets:
        convert(model, 'data/rawcap_auto_acd_vggsound.csv', args.out/'capembs_auto_acd_vggsound.npy', args.bs)
    if 'W' in args.datasets:
        convert(model, 'data/rawcap_wav_caps.csv', args.out/'capembs_wav_caps.npy', args.bs)
    if 'S' in args.datasets:
        convert(model, 'data/rawcap_sound_ve_caps.csv', args.out/'capembs_sound_ve_caps.npy', args.bs)
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)

