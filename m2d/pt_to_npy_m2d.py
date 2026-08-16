"""
Transcode IndicJEPA's cached log-mel tensors (.pt, fp16, shape (128, T)) into the
.npy (fp32) format M2D's train_audio.py loads, preserving relative paths, and write
the CSV M2D reads (file_name column, paths relative to --npy_root).

No mel re-extraction: the spectrogram values are IndicJEPA's own, so M2D trains on a
byte-identical front-end. Only the container (.pt->.npy) and dtype (fp16->fp32) change.

    python pt_to_npy_m2d.py \
        --pt_root  /home/pranav/Projects/indicJEPA/cache/<indicvoices_pretrain_dir> \
        --npy_root /home/pranav/Projects/indicJEPA/m2d/data/indicvoices_lms \
        --csv_out  /home/pranav/Projects/indicJEPA/m2d/data/files_indicvoices.csv
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_root", required=True, help="dir of IndicJEPA .pt mel cache (pretraining split)")
    ap.add_argument("--npy_root", required=True, help="output dir for M2D .npy files")
    ap.add_argument("--csv_out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="transcode only N files (smoke test); 0 = all")
    args = ap.parse_args()

    pt_root = Path(args.pt_root)
    npy_root = Path(args.npy_root)
    files = sorted(pt_root.rglob("*.pt"))
    assert files, f"no .pt under {pt_root}"
    if args.limit:
        files = files[: args.limit]

    npy_root.mkdir(parents=True, exist_ok=True)
    rows = []
    bad = 0
    for i, f in enumerate(files):
        try:
            a = torch.load(f, weights_only=True).float()          # fp16 -> fp32
        except Exception as e:
            print(f"skip {f}: {e}")
            bad += 1
            continue
        a = a.numpy().astype(np.float32)
        if a.ndim == 3 and a.shape[0] == 1:                        # (1,128,T) -> (128,T) just in case
            a = a[0]
        assert a.ndim == 2 and a.shape[0] == 128, f"{f}: unexpected shape {a.shape}"
        rel = f.relative_to(pt_root).with_suffix(".npy")
        dst = npy_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, a)
        rows.append(str(rel))
        if (i + 1) % 10000 == 0:
            print(f"{i + 1}/{len(files)}")

    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w") as fh:
        fh.write("file_name\n")
        for r in rows:
            fh.write(r + "\n")

    print(f"transcoded {len(rows)} files ({bad} skipped) -> {npy_root}")
    print(f"CSV -> {args.csv_out}")
    print(f"M2D flags:  --data_path {npy_root}  --csv_main {args.csv_out}  --input_size 128x256 --patch_size 16x16")


if __name__ == "__main__":
    main()
