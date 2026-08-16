# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from util.analyze_repr import get_cumulative_explained_variance_auc, rank_me
from einops import rearrange


def evaluate_feature(data_loader_train, data_loader_val, device, model, n_samples=1000):
    embs = []
    with torch.no_grad():
        for batch in data_loader_train:  # sample only or (mixed, clean) or (sample, label)
            if isinstance(batch, (list, tuple)):
                if batch[1].shape == batch[0].shape:
                    batch = batch[1]  # clean from (mixed, clean)
                else:
                    batch = batch[0]  # sample from (sample, label)
            sample = batch.to(device, non_blocking=True)
            B = sample.shape[0]
            _model = model if hasattr(model, 'forward_encoder') else model.module
            x, *_ = _model.forward_encoder(sample, mask_ratio=0.0)

            x = x[:, 1:]   # remove CLS
            x = rearrange(x, 'b l d -> (b l) d')
            embs.append(x)
            if B * len(embs) > n_samples:
                break # print(f'Stopped loading samples ({B*len(embs)}) > {n_samples}.')
    embs = torch.vstack(embs)
    n_actual_features = embs.shape[0]
    rank = rank_me(embs).cpu()
    auc = get_cumulative_explained_variance_auc(embs).cpu()

    val_message, val_rank, val_auc, embs = '', -1., -1., []
    if data_loader_val is not None:
        with torch.no_grad():
            for sample in data_loader_val:
                sample = sample.to(device, non_blocking=True)
                x, *_ = _model.forward_encoder(sample, mask_ratio=0.0)

                x = x[:, 1:]   # remove CLS
                x = rearrange(x, 'b l d -> (b l) d')
                embs.append(x)
                if B * len(embs) > n_samples:
                    break # print(f'Stopped loading samples ({B*len(embs)}) > {n_samples}.')
        embs = torch.vstack(embs)
        n_val_features = embs.shape[0]
        val_rank = rank_me(embs).cpu()
        val_auc = get_cumulative_explained_variance_auc(embs).cpu()
        val_message = f', val_rank={val_rank:.1f} val_auc={val_auc:.5f} ({n_val_features} features)'

    print(f'effective rank={rank:.1f} train_auc={auc:.5f} ({n_actual_features} features)' + val_message)
    return rank, auc, val_rank, val_auc


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, ema_scheduler,
                    log_writer=None,
                    val_loader: Iterable=None,
                    do_analysis: bool=False,
                    autocast_args: dict={},
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ema_decay', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if isinstance(samples, (list, tuple)):
            (samples, targs) = samples
        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(**autocast_args):
            if args.model.startswith('m2d_vit'):
                loss, *_ = model(samples, mask_ratio=args.mask_ratio)
            elif args.model.startswith('m2d_s_vit'):
                targs = targs.to(device, non_blocking=True)
                loss, _, (_, _, loss_online, loss_offline) = model(samples, targs, mask_ratio=args.mask_ratio)
            else:
                assert False, f'Unknown model: {args.model}'

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        is_update_grad = (data_iter_step + 1) % accum_iter == 0
        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=args.clip_grad, update_grad=is_update_grad)
        if is_update_grad:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Update target network
        global_step = len(data_loader) * epoch + data_iter_step
        ema_decay = ema_scheduler(global_step)
        if is_update_grad:
            _model = model if hasattr(model, 'update_target_network') else model.module
            _model.update_target_network(ema_decay)

        metric_logger.update(loss=loss_value)
        if args.model.startswith('m2d_s_vit'):
            metric_logger.update(l_on=loss_online, l_off=loss_offline)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(ema_decay=ema_decay)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('ema_decay', ema_decay, epoch_1000x)

    # feature analysis
    rank, auc, val_rank, val_auc = evaluate_feature(data_loader, val_loader, device, model) if do_analysis else (-1., -1., -1., -1.)
    metric_logger.update(rank=float(rank), auc=float(auc))
    if val_loader is not None:
        metric_logger.update(val_rank=float(val_rank), val_auc=float(val_auc))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# The original MAE training loop.
def train_one_epoch_mae(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    val_loader: Iterable=None,
                    do_analysis: bool=False,
                    autocast_args: dict={},
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if isinstance(samples, (list, tuple)):
            (samples, targs) = samples
        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(**autocast_args):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    clip_grad=args.clip_grad,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # validation loss
    if val_loader is not None:
        model.eval()
        N_TOTAL_VAL = 2048  # Total number of validation samples -- keep it small to train faster.
        N_DL = len(val_loader)
        N_VAL = max(1, (N_TOTAL_VAL + N_DL - 1)//N_DL)
        for sample in val_loader:
            sample = sample[torch.randperm(sample.shape[0])[:N_VAL]]  # random sample N_VAL items.
            sample = sample.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(**autocast_args):
                with torch.no_grad():
                    loss, *_ = model(sample, mask_ratio=args.mask_ratio)
                    metric_logger.update(val_loss=loss.item())

    # feature analysis
    rank, auc, val_rank, val_auc = evaluate_feature(data_loader, val_loader, device, model) if do_analysis else (-1., -1., -1., -1.)
    metric_logger.update(rank=float(rank), auc=float(auc))
    if val_loader is not None:
        metric_logger.update(val_rank=float(val_rank), val_auc=float(val_auc))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_m2dx(model: torch.nn.Module, teacher_model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, ema_scheduler,
                    log_writer=None,
                    val_loader: Iterable=None,
                    do_analysis: bool=False,
                    autocast_args: dict={},
                    args=None):
    """M2D-X training loop."""
    if teacher_model is not None: teacher_model.eval()
    model.train(True)
    _model = model if hasattr(model, 'update_target_network') else model.module
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ema_decay', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    use_offline_target = model.use_offline_target if hasattr(model, 'use_offline_target') else model.module.use_offline_target
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x_online, x_target = samples[0].to(device, non_blocking=True), samples[1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(**autocast_args):
            if teacher_model is None:
                loss, *others = model(x_online, x_target, mask_ratio=args.mask_ratio) if use_offline_target else model(x_online, mask_ratio=args.mask_ratio)
                try:
                    _, (_, _, loss_online, loss_offline) = others
                except:
                    loss_online, loss_offline = loss, None
            else:
                with torch.no_grad():
                    z_target = teacher_model.encode_lms(x_target)
                loss, _, (_, _, loss_online, loss_offline) = model(x_online, z_target, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        is_update_grad = (data_iter_step + 1) % accum_iter == 0
        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=args.clip_grad, update_grad=is_update_grad)
        if is_update_grad:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Update target network
        global_step = len(data_loader) * epoch + data_iter_step
        ema_decay = ema_scheduler(global_step)
        if is_update_grad:
            _model.update_target_network(ema_decay)
            # Clip logit scale if a CLAP model owns it
            _clip_logit_scale = getattr(_model, "clip_logit_scale", None)
            if callable(_clip_logit_scale):
                _clip_logit_scale()

        metric_logger.update(loss=loss_value)
        if loss_offline is not None:
            metric_logger.update(l_on=loss_online, l_off=loss_offline)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(ema_decay=ema_decay)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('ema_decay', ema_decay, epoch_1000x)

    # validation loss
    if val_loader is not None:
        if teacher_model is not None: teacher_model.eval()
        model.eval()
        N_TOTAL_VAL = 2048  # Total number of validation samples -- keep it small to train faster.
        N_DL = len(val_loader)
        N_VAL = max(1, (N_TOTAL_VAL + N_DL - 1)//N_DL)
        for sample in val_loader:
            sample = sample[torch.randperm(sample.shape[0])[:N_VAL]]  # random sample N_VAL items.
            sample = sample.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(**autocast_args):
                with torch.no_grad():
                    if teacher_model is None:
                        loss, *others = model(sample, None, mask_ratio=args.mask_ratio) if use_offline_target else model(sample, mask_ratio=args.mask_ratio)
                        try:
                            _, (_, _, loss_online, loss_offline) = others
                            metric_logger.update(val_loss=loss.item(), val_l_on = loss_online.item(), val_l_off=loss_offline.item())
                        except:
                            metric_logger.update(val_loss=loss.item())
                    else:
                        z_target = teacher_model.encode_lms(sample)
                        loss, _, (_, _, loss_online, loss_offline) = model(sample, z_target, mask_ratio=args.mask_ratio)
                        metric_logger.update(val_loss=loss.item(), val_l_on = loss_online.item(), val_l_off=loss_offline.item())

    # feature analysis
    rank, auc, val_rank, val_auc = evaluate_feature(data_loader, val_loader, device, model) if do_analysis else (-1., -1., -1., -1.)
    metric_logger.update(rank=float(rank), auc=float(auc))
    if val_loader is not None:
        metric_logger.update(val_rank=float(val_rank), val_auc=float(val_auc))

    # for CLIP model: logit_scale
    _logit_scale = getattr(_model, "logit_scale", None)
    if _logit_scale is not None:
        metric_logger.update(logit_scale=float(_logit_scale.detach().cpu()))

    if hasattr(_model, 'loss_m2d'):
        metric_logger.update(lamda_m2d=_model.loss_m2d, lamda_off=_model.loss_off)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
