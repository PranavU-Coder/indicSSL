from __future__ import annotations
import argparse, os, sys, math, time, random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.environ.get("INDICJEPA_ROOT", os.path.expanduser("~/Projects/indicJEPA"))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "indicJEPA"))

from src.indicJEPA.mae import build_from_config           # noqa: E402
from src.indicJEPA.ajepa import MelDataset          # noqa: E402


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def cosine_lr(step, base_lr, warmup, total, start_lr, min_lr=0.0):
    if step < warmup:
        return start_lr + (base_lr - start_lr) * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    d, sch, opt_c, tr = cfg["data"], cfg["schedule"], cfg["optimizer"], cfg["training"]
    mask = cfg["masking"]
    seed = int(tr["seed"])
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)
    run_name = cfg["logging"].get("run_name") or f"indicmae_seed{seed}"
    print(f"== IndicMAE reconstruction | run={run_name} | seed={seed} | device={device}")

    cache = d["dataset_path"]
    ds = MelDataset(cache, "dev", n_frames=int(d["n_time_bins"]))
    dl = DataLoader(ds, batch_size=int(d["batch_size"]), shuffle=True,
                    num_workers=int(d["num_workers"]), drop_last=True, pin_memory=True)
    print(f"   dataset: {len(ds)} clips, {len(dl)} batches/epoch")

    model = build_from_config(cfg).to(device)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"   params: encoder {n_enc/1e6:.2f}M  decoder {n_dec/1e6:.2f}M  "
          f"(target: 21.44M / 2.82M)")

    opt = torch.optim.AdamW(model.parameters(), lr=float(opt_c["lr"]),
                            weight_decay=float(opt_c["weight_decay"]),
                            betas=tuple(opt_c["betas"]))
    grad_clip = float(opt_c.get("gradient_clip", 0)) or None
    use_bf16 = tr.get("precision", "fp32") == "bf16"

    total_steps = int(sch["total_steps"])
    warmup = int(sch["warmup_steps"])
    base_lr = float(opt_c["lr"])
    start_lr = float(sch["warmup_start_lr"])
    ckpt_every = int(tr["checkpoint_every"])
    log_every = int(tr["log_every"])
    mr_min, mr_max = float(mask["mask_ratio_min"]), float(mask["mask_ratio_max"])

    step = 0
    model.train()
    t0 = time.time()
    while step < total_steps:
        for imgs in dl:
            if step >= total_steps:
                break
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(1)
            imgs = imgs.to(device, non_blocking=True)

            lr = cosine_lr(step, base_lr, warmup, total_steps, start_lr)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(imgs, mr_min, mr_max, norm_pix_loss=False)
            else:
                loss = model(imgs, mr_min, mr_max, norm_pix_loss=False)

            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            if step % log_every == 0:
                rate = (step + 1) / (time.time() - t0)
                print(f"   step {step:6d}/{total_steps}  loss {loss.item():.4f}  "
                      f"lr {lr:.2e}  {rate:.1f} it/s")

            if step > 0 and step % ckpt_every == 0:
                path = os.path.join(out_dir, f"{run_name}_step{step}.pt")
                torch.save({"model": model.state_dict(), "step": step, "config": cfg}, path)

            step += 1

    final = os.path.join(out_dir, f"{run_name}_final.pt")
    torch.save({"model": model.state_dict(), "step": step, "config": cfg}, final)
    print(f"== done. final -> {final}")

    prefixes = sorted(set(k.split(".")[0] for k in model.state_dict()))
    print(f"   key prefixes: {prefixes}  (must be ['decoder','encoder'])")
    n_enc_k = sum(1 for k in model.state_dict() if k.startswith("encoder"))
    n_dec_k = sum(1 for k in model.state_dict() if k.startswith("decoder"))
    print(f"   tensor counts: encoder {n_enc_k}, decoder {n_dec_k}  (s42: 149, 80)")


if __name__ == "__main__":
    main()