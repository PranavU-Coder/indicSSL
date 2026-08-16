"""
s3prl custom upstream for IndicMAE (reconstruction, ViT-on-mel).

Drop-in replacement for  s3prl/upstream/example/expert.py  -> run with  -u customized_upstream.
Evaluates the FROZEN target encoder (the module the paper probes), under the SUPERB protocol.

Why this file exists (see paper's eval-setup paragraph): Audio-JEPA encodes a fixed 2D grid of
ViT patches, not a duration-proportional frame stream. s3prl's Featurizer computes each clip's
length as round(len(wav) / downsample_rate) and pools frame-level heads over that axis, so the
upstream must return a *time-ordered* sequence whose length scales with duration. We therefore
  (i)  collapse the frequency-patch axis (mean over the 8 freq patches per time step) to recover
       a temporal sequence -- for mean-pooled tasks this equals eval.py's pool over all 128 tokens;
  (ii) window each utterance into fixed clip_duration chunks (the encoder is fixed-input: its
       positional buffer is sized to exactly grid_h*grid_w patches), then trim to the real
       (non-padding) length so the returned T matches s3prl's length bookkeeping.

hidden_states = the depth transformer-block outputs (pre-final-norm); s3prl learns the layer weights.

Setup (in the IndicSUPERB venv):
    export INDICJEPA_ROOT=/home/pranav/Projects/indicJEPA   # repo root, so `src.indicJEPA.*` imports
    cp expert.py <IndicSUPERB>/s3prl/upstream/example/expert.py
Run:
    python s3prl/run_downstream.py -n ijepa_lid -m train -u customized_upstream \
        -k /home/pranav/Projects/indicJEPA/runs/indicjepa_s42/final.pt -d indic_lid \
        -o "config.downstream_expert.datarc.file_path=<root>/kb_data_clean_wav"
"""

import os
import sys
import math
from typing import List, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

# make `src.indicJEPA.ajepa` importable from the eval venv
_ROOT = os.environ.get("INDICJEPA_ROOT")
if _ROOT and _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.indicJEPA.ajepa import ViTEncoder  # noqa: E402


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        super().__init__()
        assert ckpt is not None, "pass the checkpoint via -k <path/to/final.pt>"

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = state["config"]
        m = cfg["model"]
        self.encoder = ViTEncoder(
            n_mels=int(cfg["data"]["n_mels"]),
            n_frames=int(cfg["data"]["n_time_bins"]),
            patch_size=int(m["patch_size"]),
            embed_dim=int(m["embed_dim"]),
            depth=int(m["depth"]),
            num_heads=int(m["num_heads"]),
            mlp_ratio=float(m.get("mlp_ratio", 4.0)),
        )
        sd = state["model"]
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=True)
        assert not missing and not unexpected, \
            f"MAE encoder mismatch: missing={missing[:4]} unexpected={unexpected[:4]}"
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # ---- front-end: replicate prep.py / indic.py EXACTLY from the same config ----
        d = cfg["data"]
        sr = int(d["sample_rate"])              # 16000
        dur = float(d["clip_duration"])         # 5.0
        self.n_mels = int(d["n_mels"])          # 128
        self.n_time = int(d["n_time_bins"])     # 256
        ratio = float(d["frame_to_hop_ratio"])  # 2.5
        self.win_samples = int(sr * dur)                 # 80000  (one fixed-input clip)
        self.hop = self.win_samples // self.n_time        # 312
        n_fft = int(round(self.hop * ratio))              # 780

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=self.hop, n_mels=self.n_mels,
            f_min=50, f_max=sr // 2, power=2.0,
        )

        # patch grid (single source of truth = the encoder)
        self.grid_h = self.encoder.grid_h                 # 8  (frequency patches)
        self.grid_w = self.encoder.grid_w                 # 16 (time patches per window)
        self.n_layers = len(self.encoder.blocks)          # 12
        self.embed_dim = self.encoder.embed_dim

        # samples per output (time) token, after the freq-collapse
        self._downsample = self.hop * (self.n_time // self.grid_w)   # 312 * 16 = 4992

    def get_downsample_rates(self, key: str) -> int:
        return self._downsample

    def _logmel_windows(self, wav: Tensor) -> Tensor:
        """wav (L,) -> (n_win, 1, n_mels, n_time), padded to whole windows, per-window normed."""
        L = wav.shape[-1]
        n_win = max(1, math.ceil(L / self.win_samples))
        pad_to = n_win * self.win_samples
        if L < pad_to:
            wav = F.pad(wav, (0, pad_to - L))
        windows = wav.reshape(n_win, self.win_samples)             # (n_win, win_samples)

        mel = torch.log(self.mel(windows) + 1e-6)                  # (n_win, n_mels, T~257)
        T = mel.shape[-1]
        if T >= self.n_time:
            mel = mel[..., : self.n_time]
        else:
            mel = F.pad(mel, (0, self.n_time - T), value=float(mel.min()))
        # per-window (== per-clip) standardization, matching FSD50KMelDataset / eval.py
        m = mel.mean(dim=(-2, -1), keepdim=True)
        s = mel.std(dim=(-2, -1), keepdim=True)
        mel = (mel - m) / (s + 1e-6)
        return mel.unsqueeze(1)                                    # (n_win, 1, n_mels, n_time)

    @torch.no_grad()
    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        device = next(self.encoder.parameters()).device
        per_layer: List[List[Tensor]] = [[] for _ in range(self.n_layers)]

        for wav in wavs:
            wav = wav.to(device).float()
            if wav.dim() > 1:
                wav = wav.reshape(-1)
            L = wav.shape[-1]

            x = self._logmel_windows(wav)                         # (n_win, 1, mel, time)
            n_win = x.shape[0]
            feats = self.encoder.forward_intermediate(x)          # list[n_layers] of (n_win, N, D)

            # real (non-padding) time-token count for this clip
            keep = max(1, min(round(L / self._downsample), n_win * self.grid_w))

            for li, f in enumerate(feats):
                # (n_win, N=grid_h*grid_w, D) -> collapse frequency -> time-ordered sequence
                f = f.view(n_win, self.grid_h, self.grid_w, self.embed_dim).mean(dim=1)  # (n_win, grid_w, D)
                f = f.reshape(n_win * self.grid_w, self.embed_dim)[:keep]                # (keep, D)
                per_layer[li].append(f)

        # pad across the batch -> each layer (B, T_max, D); T_max = max keep = round(max_len/rate)
        hidden_states = [pad_sequence(layer, batch_first=True) for layer in per_layer]
        return {"hidden_states": hidden_states}
