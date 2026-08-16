"""
Standalone check for the s3prl upstream wrapper (expert.py) -- no s3prl task needed.
Validates shapes, the layer set, NaNs, and the T == round(len(wav)/downsample) length
contract that s3prl's Featurizer asserts.

Run from the IndicSUPERB dir:
    export INDICJEPA_ROOT=~/Projects/indicJEPA
    python smoke_wrapper.py \
        --ckpt ~/Projects/indicJEPA/runs/indicjepa_s42/final.pt \
        --wav_dir ~/Projects/indicJEPA/data/kb_data_clean_wav/hindi/valid
"""
import argparse
import glob
import sys
from pathlib import Path

import torch
import soundfile as sf

sys.path.insert(0, str(Path("s3prl/upstream/example").resolve()))
from expert import UpstreamExpert  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    up = UpstreamExpert(ckpt=args.ckpt).to(device).eval()
    rate = up.get_downsample_rates("hidden_states")
    print(f"device={device}  downsample_rate={rate}")

    files = sorted(glob.glob(f"{args.wav_dir}/**/*.wav", recursive=True))[: args.n]
    assert files, f"no wavs under {args.wav_dir}"

    wavs = []
    for f in files:
        arr, sr = sf.read(f, dtype="float32", always_2d=False)
        assert sr == 16000, f"{Path(f).name}: expected 16 kHz, got {sr}"
        w = torch.from_numpy(arr)
        if w.ndim > 1:
            w = w.mean(-1)
        wavs.append(w.to(device))
        print(f"  {Path(f).name}: {w.shape[-1]:>7d} samples ({w.shape[-1] / 16000:.2f}s)")

    out = up(wavs)
    hs = out["hidden_states"]
    x = hs[0]
    max_len = max(w.shape[-1] for w in wavs)
    exp_T = round(max_len / rate)

    print("\n--- checks ---")
    print(f"num layers            : {len(hs)}          (expect 12)")
    print(f"layer0 shape (B,T,D)  : {tuple(x.shape)}")
    print(f"D == 384              : {x.shape[-1] == 384}")
    print(f"T == round(maxlen/ds) : {x.shape[1] == exp_T}   (T={x.shape[1]}, expected {exp_T})")
    print(f"all layers same shape : {all(h.shape == x.shape for h in hs)}")
    print(f"any NaN               : {any(torch.isnan(h).any().item() for h in hs)}")
    print(f"any Inf               : {any(torch.isinf(h).any().item() for h in hs)}")
    print(f"layer0 mean/std       : {x.float().mean():.3f} / {x.float().std():.3f}")

    ok = (
        len(hs) == 12
        and x.shape[-1] == 384
        and x.shape[1] == exp_T
        and all(h.shape == x.shape for h in hs)
        and not any(torch.isnan(h).any().item() for h in hs)
    )
    print("\nRESULT:", "PASS wrapper is sound" if ok else "FAIL see checks above")


if __name__ == "__main__":
    main()
