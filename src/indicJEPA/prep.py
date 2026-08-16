"""
preprocess wavs into cached log-mel-spectrograms.
reads mel-spec parameters from the YAML config so they stay consistent with
training. Computes n_fft and hop_length from sample_rate, clip_duration,
n_time_bins, and frame_to_hop_ratio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import torch
import torchaudio
from tqdm import tqdm

from configs import load_config


def compute_mel_params(cfg):
    """compute (n_fft, hop_length) from data config.

    solves:  n_time_bins = sample_rate * clip_duration / hop_length
             n_fft       = round(hop_length * frame_to_hop_ratio)
    """
    sr = cfg["data"]["sample_rate"]
    dur = cfg["data"]["clip_duration"]
    n_time = cfg["data"]["n_time_bins"]
    ratio = cfg["data"]["frame_to_hop_ratio"]
    total_samples = int(sr * dur)
    hop_length = total_samples // n_time
    n_fft = int(round(hop_length * ratio))
    return n_fft, hop_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fsd_root",
        required=True,
        help="FSD50K root (contains FSD50K.dev_audio/ etc.)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="where to save cached log-mel tensors"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--splits", nargs="+", default=["dev"], choices=["dev", "eval"])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sr = cfg["data"]["sample_rate"]
    n_mels = cfg["data"]["n_mels"]
    n_fft, hop_length = compute_mel_params(cfg)

    expected_frames = int(sr * cfg["data"]["clip_duration"]) // hop_length
    print(f"mel params: sr={sr}, n_mels={n_mels}, n_fft={n_fft}, hop={hop_length}")
    print(
        f": ~{expected_frames} frames per {cfg['data']['clip_duration']}s clip "
        f"(target n_time_bins={cfg['data']['n_time_bins']})"
    )

    device = torch.device(args.device)
    mel_xform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=50,
        f_max=sr // 2,
        power=2.0,
    ).to(device)

    fsd_root = Path(args.fsd_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        src = fsd_root / f"FSD50K.{split}_audio"
        dst = out_root / split
        dst.mkdir(exist_ok=True)
        wavs = sorted(src.glob("*.wav"))
        if not wavs:
            print(f"no wavs in {src} or skipping {split}.")
            continue
        print(f"[{split}] processing {len(wavs)} files into {dst}")

        n_done = n_skip = n_err = 0
        for wav_path in tqdm(wavs, dynamic_ncols=True):
            out_path = dst / (wav_path.stem + ".pt")
            if out_path.exists():
                n_skip += 1
                continue
            try:
                wav, file_sr = torchaudio.load(str(wav_path))
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if file_sr != sr:
                    wav = torchaudio.functional.resample(wav, file_sr, sr)
                wav = wav.to(device)
                mel = mel_xform(wav)
                log_mel = torch.log(mel + 1e-6)
                log_mel = log_mel.squeeze(0).half().cpu()
                torch.save(log_mel, out_path)
                n_done += 1
            except Exception as e:
                print(f"! {wav_path.name}: {e}")
                n_err += 1

        print(f"[{split}] new: {n_done}, already cached: {n_skip}, errors: {n_err}")

    print("done.")


if __name__ == "__main__":
    main()
