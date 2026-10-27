import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import io

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset
from tqdm import tqdm

from configs import load_config
from src.indicJEPA.prep import compute_mel_params

IV_LANGS_12 = [
    "hindi", "bengali", "marathi", "tamil", "telugu", "kannada",
    "malayalam", "gujarati", "punjabi", "odia", "urdu", "sanskrit",
]

AUDIO_KEYS = ("audio_filepath", "audio")


def _decode(a):
    """`a` is an Audio(decode=False) value: {'bytes': ..., 'path': ...}."""
    src = io.BytesIO(a["bytes"]) if a.get("bytes") is not None else a["path"]
    wav, sr = sf.read(src, dtype="float32", always_2d=False)
    return wav, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True,
                    help="cache root (pass this to train.py --data_dir)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="dev", help="writes into <output_dir>/<split>/")
    ap.add_argument("--hours_per_lang", type=float, default=5.0)
    ap.add_argument("--languages", nargs="+", default=IV_LANGS_12)
    ap.add_argument("--hf_split", default="train", help="IndicVoices split to stream")
    ap.add_argument("--token", default=None, help="HF token (or just huggingface-cli login)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sr = cfg["data"]["sample_rate"]
    n_mels = cfg["data"]["n_mels"]
    n_fft, hop = compute_mel_params(cfg)
    print(f"mel params: sr={sr}  n_mels={n_mels}  n_fft={n_fft}  hop={hop}")

    device = torch.device(args.device)
    mel_xform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        f_min=50, f_max=sr // 2, power=2.0,
    ).to(device)

    for lang in args.languages:
        dst = Path(args.output_dir) / args.split
        dst.mkdir(parents=True, exist_ok=True)
        try:
            ds = load_dataset("ai4bharat/IndicVoices", lang,
                              split=args.hf_split, streaming=True, token=args.token)
        except Exception as e:
            print(f"[{lang}] skipped ({e})")
            continue

        feats = ds.features or {}
        audio_col = next((k for k in AUDIO_KEYS if k in feats), "audio_filepath")
        ds = ds.cast_column(audio_col, Audio(decode=False))

        budget_s = args.hours_per_lang * 3600.0
        got_s, idx = 0.0, 0
        pbar = tqdm(ds, desc=lang, unit="clip", dynamic_ncols=True)
        for ex in pbar:
            if got_s >= budget_s:
                break
            wav_np, file_sr = _decode(ex[audio_col])
            wav = torch.from_numpy(np.ascontiguousarray(wav_np))
            if wav.ndim > 1:
                wav = wav.mean(dim=-1)
            got_s += wav.numel() / file_sr
            wav = wav.unsqueeze(0).to(device)
            if file_sr != sr:
                wav = torchaudio.functional.resample(wav, file_sr, sr)
            mel = mel_xform(wav)
            log_mel = torch.log(mel + 1e-6).squeeze(0).half().cpu()
            torch.save(log_mel, dst / f"{lang}_{idx:07d}.pt")
            idx += 1
            pbar.set_postfix(hours=f"{got_s / 3600:.2f}")
        print(f"[{lang}] {idx} clips  (~{got_s / 3600:.2f} h)  -> {dst}")

    print("done.")


if __name__ == "__main__":
    main()