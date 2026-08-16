"""
IndicSUPERB ships no chunker: LID trains on fixed-length segments (paths containing
'chunk_data', label = parent folder) while valid/test use full utterances. This
segments the converted train wavs into chunk_s-second pieces and writes them flat.

usage:
    python chunk_lid_train.py \
        --wav_root ~/Projects/indicJEPA/data/kb_data_clean_wav \
        --langs hindi kannada --chunk_s 5.0
"""
import argparse
import glob
from pathlib import Path

import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--wav_root", required=True, help="kb_data_clean_wav")
ap.add_argument("--langs", nargs="+", required=True)
ap.add_argument("--chunk_s", type=float, default=5.0)
ap.add_argument("--min_s", type=float, default=1.0, help="drop trailing pieces shorter than this")
ap.add_argument("--sr", type=int, default=16000)
args = ap.parse_args()

clen = int(args.chunk_s * args.sr)
mlen = int(args.min_s * args.sr)

for lang in args.langs:
    src = Path(args.wav_root) / lang / "train"
    dst = Path(args.wav_root) / "chunk_data" / lang
    dst.mkdir(parents=True, exist_ok=True)
    wavs = sorted(glob.glob(str(src / "**" / "*.wav"), recursive=True))
    if not wavs:
        print(f"[{lang}] no train wavs under {src} -- convert the train split first")
        continue
    n = 0
    for w in wavs:
        arr, sr = sf.read(w, dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(-1)
        stem = Path(w).stem
        for i in range(0, len(arr), clen):
            seg = arr[i : i + clen]
            if len(seg) < mlen:
                continue
            sf.write(str(dst / f"{stem}_{i // clen:03d}.wav"), seg, sr)
            n += 1
    print(f"[{lang}] {len(wavs)} utts -> {n} chunks -> {dst}")
