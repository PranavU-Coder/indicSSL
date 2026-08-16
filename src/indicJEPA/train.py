""" usage:

    # smoke test
    python src/indicJEPA/train.py --config configs/default.yaml \
        --data_dir ./cache/indicvoices --output_dir ./runs/smoke \
        --override training.max_steps=500

    # main baseline run
    python src/indicJEPA/train.py --config configs/default.yaml \
        --data_dir ./cache/indicvoices --output_dir ./runs/indicjepa_s42 \
        --override training.seed=42

    # resume
    python src/indicJEPA/train.py --config configs/default.yaml \
        --data_dir ./cache/indicvoices --output_dir ./runs/indicjepa_s42 \
        --resume ./runs/indicjepa_s42/ckpt_step20000.pt
"""

import sys
import argparse
import math
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.indicJEPA.ajepa import FSD50KMelDataset, AudioJEPA
from configs import load_config, apply_overrides


def cosine(step, total, base, final, warmup=0, warm_start=0.0):
    """cosine schedule with optional linear warmup."""
    if step < warmup:
        return warm_start + (base - warm_start) * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * p))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--data_dir", required=True, help="output of prep.py (contains dev/ etc.)"
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="key.subkey=value overrides for the config",
    )
    p.add_argument(
        "--resume", type=str, default=None, help="path to a checkpoint to resume from"
    )
    return p.parse_args()


def maybe_init_wandb(cfg, out_dir):
    proj = cfg.get("logging", {}).get("wandb_project")
    if not proj:
        return None
    try:
        import wandb
    except ImportError:
        print(
            "wandb requested in config but not installed; skipping. "
            "pip install wandb to enable."
        )
        return None
    return wandb.init(
        project=proj,
        entity=cfg["logging"].get("wandb_entity"),
        name=cfg["logging"].get("run_name") or out_dir.name,
        config=cfg,
        dir=str(out_dir),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    torch.manual_seed(cfg["training"].get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"device:  {device}")
    print(f"config:  {args.config} -> {out_dir}/config.yaml")
    print(f"output:  {out_dir}")

    n_frames = cfg["data"]["n_time_bins"]
    ds = FSD50KMelDataset(args.data_dir, "dev", n_frames)
    loader = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg["data"]["num_workers"] > 0),
    )

    model = AudioJEPA.from_config(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params: {n_total / 1e6:.2f} M total ({n_train / 1e6:.2f} M trainable)")

    optc = cfg["optimizer"]
    optim = AdamW(
        model.trainable_parameters(),
        lr=optc["lr"],
        betas=tuple(optc["betas"]),
        weight_decay=optc["weight_decay"],
    )

    prec = cfg["training"].get("precision", "bf16")
    dtype = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }[prec]
    use_amp = dtype != torch.float32
    scaler = GradScaler("cuda") if dtype == torch.float16 else None

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resumed from {args.resume} at step {step}")

    run = maybe_init_wandb(cfg, out_dir)

    sched = cfg.get("schedule", {})
    total_steps = cfg["training"]["max_steps"]
    warmup_steps = sched.get("warmup_steps", 1000)
    warmup_start_lr = sched.get("warmup_start_lr", 1e-6)
    base_lr = optc["lr"]
    tau_start = sched.get("ema_tau_start", 0.996)
    tau_end = sched.get("ema_tau_end", 1.0)
    grad_clip = optc.get("gradient_clip", 1.0)
    log_every = cfg["training"].get("log_every", 100)
    ckpt_every = cfg["training"].get("checkpoint_every", 5000)

    log_loss = 0.0
    log_n = 0
    t0 = time.time()
    model.train()

    while step < total_steps:
        for x in loader:
            if step >= total_steps:
                break
            x = x.to(device, non_blocking=True)

            lr = cosine(
                step,
                total_steps,
                base_lr,
                0.0,
                warmup=warmup_steps,
                warm_start=warmup_start_lr,
            )
            for g in optim.param_groups:
                g["lr"] = lr
            tau = cosine(step, total_steps, tau_start, tau_end)

            optim.zero_grad(set_to_none=True)
            if use_amp:
                with autocast("cuda", dtype=dtype):
                    loss, ctx_emb = model(x)
            else:
                loss, ctx_emb = model(x)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), grad_clip
                )
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), grad_clip
                )
                optim.step()

            model.update_target(tau)

            step += 1
            log_loss += loss.item()
            log_n += 1

            if step % log_every == 0:
                avg = log_loss / log_n
                with torch.no_grad():
                    # LayerNorm forces unit std within a token, so the only meaningful collapse signal lives across samples.
                    pooled = ctx_emb.detach().float().mean(dim=1)  # (B, D)
                    sample_std = pooled.std(dim=0).mean().item()
                    pooled_n = torch.nn.functional.normalize(pooled, dim=-1)
                    cos = pooled_n @ pooled_n.T  # (B, B)
                    B = pooled.size(0)
                    avg_cos = (cos.sum() - cos.diagonal().sum()).item() / (B * (B - 1))
                sps = log_n / (time.time() - t0 + 1e-9)
                print(
                    f"step {step:6d}  loss {avg:.4f}  "
                    f"lr {lr:.2e}  tau {tau:.5f}  "
                    f"samp_std {sample_std:.3f}  cos {avg_cos:+.3f}  "
                    f"grad {grad_norm:.2f}  "
                    f"{sps:.1f} step/s"
                )
                if run is not None:
                    run.log(
                        {
                            "loss": avg,
                            "lr": lr,
                            "tau": tau,
                            "sample_std": sample_std,
                            "avg_cos": avg_cos,
                            "grad_norm": float(grad_norm),
                            "step_per_sec": sps,
                        },
                        step=step,
                    )
                log_loss = 0.0
                log_n = 0
                t0 = time.time()

            if step % ckpt_every == 0:
                ckpt_path = out_dir / f"ckpt_step{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optim.state_dict(),
                        "config": cfg,
                    },
                    ckpt_path,
                )
                print(f" has been saved {ckpt_path}")

    final = out_dir / "final.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "config": cfg,
        },
        final,
    )
    print(f"done. final checkpoint: {final}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()