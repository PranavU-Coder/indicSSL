"""Masked Modeling Duo (M2D) Pre-training Script V2

Masked Modeling Duo: Towards a Universal Audio Pre-training Framework
https://arxiv.org/abs/2404.06095
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import subprocess
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import matplotlib.pyplot as plt

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from m2d import models_mae
from m2d.engine_pretrain_m2d import train_one_epoch_m2dx
import audio_dataset
import common
from m2d.runtime_audio import RuntimeM2D


def get_args_parser():
    parser = argparse.ArgumentParser('Masked Modeling Duo (M2D) pre-training V2', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--eval_after', default=50, type=int)
    parser.add_argument('--save_freq', default=100, type=int)
    parser.add_argument('--feature_eval_freq', default=10, type=int, help='Feature label-free evaluation frequency.')
    parser.add_argument('--stop_at', default=-1, type=int)

    # Model parameters
    parser.add_argument('--model', default='m2d_vit_base', type=str, metavar='MODEL', help='Model name.')
    parser.add_argument('--decoder_depth', type=int, default=8, metavar='DD', help='Model decoder depth.')

    parser.add_argument('--input_size', default='80x608', type=str, help='images input size')
    parser.add_argument('--patch_size', default='16x16', type=str, help='patch size')
    parser.add_argument('--sr', default='16k', type=str, metavar='SR', help='Sampling rate of the input audio.')

    parser.add_argument('--mask_ratio', default=0.7, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--weighted_mask', action='store_true',
                        help='Use weighted masking based on data contents.')
    parser.add_argument('--var_weighted_mask', action='store_true',
                        help='Use weighted masking (var) based on data contents.')
    parser.add_argument('--block_mask', action='store_true',
                        help='Use block masking from data2vec 2.0.')
    parser.add_argument('--obj_cent_mask', action='store_true',
                        help='Use object centric masking based on data contents.')
    parser.add_argument('--masking_hint_k', default=2, type=int)
    parser.add_argument('--masking_start_epoch', default=50, type=int)
    parser.add_argument('--abs_hint_ratio', default=-1.0, type=float)  # set value >= 0.0 to enable absolute hint ratio

    parser.add_argument('--ema_decay_init', default=0.99995, type=float,
                        help='Initial EMA decay parameter.')
    parser.add_argument('--ema_decay', default=0.99999, type=float,
                        help='EMA decay parameter.')
    parser.add_argument('--loss_fn', default='norm_mse', type=str,
                        help='loss function: mse or norm_mse.')
    parser.add_argument('--loss_m2d', default=1., type=float, help='Loss of M2D masked prediction')
    parser.add_argument('--loss_off', default=0., type=float, help='Loss of offline target')
    parser.add_argument('--target_layers', default='', type=str,
                        help='Experimental: layers to calculate target representations.')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--teacher', default='', type=str, help='Weight path of the teacher M2D model.')

    parser.add_argument('--no_norm_pix_loss', action='store_false', dest='norm_pix_loss',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    parser.add_argument('--cont_mask', default=0, type=int,
                        help='Use random 1-d (continuous) masking scheme. 0:off, 0<:# of frames to make them continuous.')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar="NORM",
                        help="Clip gradient norm (default: None, no clipping)")
    parser.add_argument('--optim', default='adamw', type=str, help='Optimizer adam or sdg')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=3e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='data', type=str,
                        help='dataset path')
    parser.add_argument('--csv_main', default='data/files_audioset.csv', type=str,
                        help='A CSV file to list sample files in the main dataset')
    parser.add_argument('--csv_bg_noise', default='', type=str,
                        help='A CSV file to list sample files in the BG noise dataset')
    parser.add_argument('--csv_val', default='', type=str,
                        help='A CSV file to list validation sample files')
    parser.add_argument('--min_ds_size', default=10000, type=int,
                        help='Inflate the size of the smaller dataset to the desired size')
    parser.add_argument('--norm_stats', default='None', type=str,  # Will be computed runtime.
                        help='dataset normalization stats')
    parser.add_argument('--noise_ratio', default=0., type=float,
                        help='Noise mixing ratio')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

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


def ema_decay_sched(step, total_steps, ema_decay_init, ema_decay):
    interp = step / (total_steps - 1)
    tau = ema_decay_init + (ema_decay - ema_decay_init) * interp
    return tau


def get_optim(args, param_groups):
    if args.optim == 'adamw':
        return torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    elif args.optim == 'sgd':
        return torch.optim.SGD(param_groups, args.lr, momentum=0.9, weight_decay=0)
    assert False, f'Unsupported optimizer {args.optim}'


def load_model(args, model_without_ddp, optimizer, loss_scaler, delta_epoch=1, strict=True):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=strict)
        print("Resume checkpoint %s" % args.resume)
        if strict == True and 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + delta_epoch
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_val = audio_dataset.build_mixed_dataset(args)
    print(dataset_train, dataset_val)

    if args.teacher == '':
        teacher_model, off_emb_dim = None, 3840
        print(f' ** No teacher with loss_off=={args.loss_off} > 0. **')
    else:
        print(f' ** M2D-X **')
        teacher_model = RuntimeM2D(weight_file=args.teacher)
        off_emb_dim = teacher_model.cfg.feature_d
        models_mae.set_requires_grad(teacher_model, False)
        teacher_model.to(device)
        teacher_model.eval()
        print('Teacher weight =', args.teacher, 'feature_d =', off_emb_dim)
        print("Teacher = %s" % common.short_model_desc(teacher_model))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    common.PrintLogger(f'{args.log_dir}/console.txt')
    print(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # for validation loss
    if args.csv_val != '':
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        print("Sampler_val = %s" % str(sampler_val))
        data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    else:
        data_loader_val = None
    
    # define the model
    model = models_mae.__dict__[args.model](img_size=args.input_size, patch_size=args.patch_size, decoder_depth=args.decoder_depth,
        norm_pix_loss=args.norm_pix_loss, loss_type=args.loss_fn, target_layers=args.target_layers, loss_m2d=args.loss_m2d, loss_off=args.loss_off,
        off_emb_dim=off_emb_dim, norm_stats=dataset_train.norm_stats, world_rank=[num_tasks, global_rank])

    if args.weighted_mask:
        assert (args.cont_mask == 0) and (not args.var_weighted_mask) and (not args.obj_cent_mask), "Masking options conflict."
        model.set_random_weighted_mask_with_hint()
    if args.var_weighted_mask:
        assert (args.cont_mask == 0) and (not args.weighted_mask) and (not args.obj_cent_mask), "Masking options conflict."
        model.set_random_weighted_mask_var_with_hint()
    if args.obj_cent_mask:
        assert (args.cont_mask == 0) and (not args.weighted_mask) and (not args.var_weighted_mask), "Masking options conflict."
        if args.masking_start_epoch == 0:
            print(f"⚠️  Warning: Too early to start object centric masking.")
        model.set_object_centric_mask_with_hint()
    if args.block_mask:
        model.set_data2vec2_block_masking()
    if args.cont_mask > 0:
        model.set_random_1d_mask(args.cont_mask)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    org_args_lr = args.lr
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / args.eff_batch_size) if org_args_lr is None else 'base lr: not effective')
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % args.eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    try:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    except:
        print(' (for compatibility with timm) Switched add_weight_decay() to param_groups_weight_decay()')
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = get_optim(args, param_groups)
    print(optimizer)
    loss_scaler = NativeScaler()

    load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, delta_epoch=0, strict=False)

    if args.force_start_epoch >= 0:
        args.start_epoch = args.force_start_epoch

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    last_subprocess = None
    for epoch in range(args.start_epoch, args.epochs):
        epoch1 = epoch + 1
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if args.weighted_mask or args.var_weighted_mask or args.block_mask or args.obj_cent_mask:
            if args.abs_hint_ratio < 0.0:
                model_without_ddp.set_hint_ratio_by_epoch(cur_epoch=epoch, total_epochs=args.epochs, start_epoch=args.masking_start_epoch, k=args.masking_hint_k)
            else:
                model_without_ddp.set_hint_ratio(args.abs_hint_ratio)
        train_stats = train_one_epoch_m2dx(
            model, teacher_model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            partial(ema_decay_sched, total_steps=len(data_loader_train) * args.epochs,
                ema_decay_init=args.ema_decay_init, ema_decay=args.ema_decay),
            val_loader=data_loader_val,
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
        subprocess.call(['/bin/bash', './all_eval.sh', abspath])
    return epoch1


arg_conf_defaults = {
    'csv_main': ('data/files_audioset.csv', 'M', 'path'),
    'csv_bg_noise': ('', 'D', 'path'),
    'ema_decay_init': (0.99995, 'ema', 'z'),
    'ema_decay': (0.99999, 'ed', 'z'),
    'decoder_depth': (8, 'dd', 'asis'),
    'mask_ratio': (0.7, 'mr', 'z'),
    'seed': (0, 's', 'asis'),
    'norm_pix_loss':  (True, '~N', 'b'),
    'loss_fn': ('norm_mse', 'L', 'head'),
    'optim': ('adamw', 'O', 'asis'),
    'warmup_epochs': (20, 'wu', 'asis'),
    'blr': (3e-4, 'blr', 'z'),
    'lr': (None, 'lr', 'z'),
    'target_layers': ('', 'T', 'b'),
    'eff_batch_size': (2048, 'bs', 'asis'),
    'accum_iter': (1, 'a', 'asis'),
    'loss_m2d': (1.0, 'lm', 'z'),
    'loss_off': (0.0, 'lo', 'z'),
    'noise_ratio': (0.0, 'nr', 'z'),
    'min_ds_size': (10000, 'dn', 'asis'),
    'cont_mask': (0, 'C', 'asis'),
    'weighted_mask': (False, 'Wm', 'b'),
    'var_weighted_mask': (False, 'Vm', 'b'),
    'block_mask': (False, 'Bm', 'b'),
    'obj_cent_mask': (False, 'Om', 'b'),
    'masking_hint_k': [2, 'k', 'asis'],
    'masking_start_epoch': [50, 'i', 'asis'],
    'abs_hint_ratio': [-1.0, 'a', 'asis'],
    'epochs': (0, '-e', 'asis'),
}


def complete_args():
    args = get_args_parser()
    args = args.parse_args()
    _input_size, _patch_size = args.input_size, args.patch_size
    args.input_size = [int(x) for x in args.input_size.split('x')]
    args.patch_size = [int(x) for x in args.patch_size.split('x')]
    args.norm_stats = eval(args.norm_stats) if args.norm_stats else None

    misc.init_distributed_mode(args)

    args.eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if not args.output_dir:
        args.output_dir = f'{args.model}-{_input_size}p{_patch_size}p{args.sr}'
        args.output_dir += f'-{common.get_timestamp()[:6]}-{common.arg_conf_str(args, defaults=arg_conf_defaults)}'

    if not args.log_dir:
        args.log_dir = args.output_dir
    args.target_layers = None if args.target_layers == '' else eval(args.target_layers)
    return args


if __name__ == '__main__':
    args = complete_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
