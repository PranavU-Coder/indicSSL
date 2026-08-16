"""
Run:
    cp expert_m2d.py <IndicSUPERB>/s3prl/upstream/example/expert.py
    python s3prl/run_downstream.py -n m2d_lid12_s42 -m train -u customized_upstream \
        -k <m2d>/m2d_vit_small-128x256p16x16p16k-260722-Mdfis42bs64-e100/checkpoint-100.pth \
        -d indic_lid -c s3prl/downstream/indic_lid/config.yaml \
        -o "config.downstream_expert.datarc.file_path=<root>/kb_data_clean_wav,,config.runner.total_steps=50000,,config.downstream_expert.datarc.num_workers=16"
"""
import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

# constants
SR = 16000
CLIP_DURATION = 5.0
N_MELS = 128
N_TIME = 256
FRAME_TO_HOP_RATIO = 2.5
F_MIN = 50

FREQ_MAJOR = True


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)      # scale = head_dim ** -0.5, as timm
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()                              # approximate='none', as timm default
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """timm ViT Block with LayerScale/DropPath as identity"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class M2DOnlineEncoder(nn.Module):
    """M2D's online encoder: patch_embed -> +pos_embed -> [cls] -> 12 blocks. No masking."""

    def __init__(self, embed_dim: int, depth: int, num_heads: int, grid_h: int, grid_w: int):
        super().__init__()
        self.embed_dim, self.grid_h, self.grid_w = embed_dim, grid_h, grid_w
        self.patch_embed_proj = nn.Conv2d(1, embed_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_h * grid_w + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])

    def forward_intermediate(self, x: Tensor) -> List[Tensor]:
        """x (B, 1, n_mels, n_time) -> list[depth] of (B, grid_h*grid_w, D), cls dropped."""
        x = self.patch_embed_proj(x)                      # (B, D, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)                  # (B, N, D), index = h*grid_w + w
        x = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)                    # (B, 1+N, D)
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x[:, 1:, :])                     # drop cls; pre-final-norm, as expert.py
        return feats


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        super().__init__()
        assert ckpt is not None, "pass the M2D checkpoint via -k <.../checkpoint-100.pth>"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = state["model"]
        args = state.get("args", None)

        embed_dim = sd["patch_embed.proj.weight"].shape[0]                    # 384
        depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        num_heads = embed_dim // 64                                          # ViT-S: 384/64 = 6
        n_patches = sd["pos_embed"].shape[1] - 1                             # 128
        self.grid_h = N_MELS // 16
        self.grid_w = n_patches // self.grid_h
        assert self.grid_h * self.grid_w == n_patches, "patch grid does not match pos_embed"
        if args is not None:
            assert list(args.input_size) == [N_MELS, N_TIME], (
                f"checkpoint input_size {args.input_size} != front-end {[N_MELS, N_TIME]}")

        self.encoder = M2DOnlineEncoder(embed_dim, depth, num_heads, self.grid_h, self.grid_w)

        remap = {"patch_embed.proj.weight": "patch_embed_proj.weight",
                 "patch_embed.proj.bias": "patch_embed_proj.bias"}
        enc_sd = {}
        for k, v in sd.items():
            if k in remap:
                enc_sd[remap[k]] = v
            elif k in ("cls_token", "pos_embed") or k.startswith("blocks."):
                enc_sd[k] = v
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        assert not missing, f"missing encoder weights: {missing[:8]}"
        assert not unexpected, f"unexpected weights: {unexpected[:8]}"

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        ns = sd["norm_stats"].float().reshape(-1)
        self.register_buffer("norm_mean", ns[0].clone())
        self.register_buffer("norm_std", ns[1].clone())

        self.n_mels, self.n_time = N_MELS, N_TIME
        self.win_samples = int(SR * CLIP_DURATION)                   # 80000
        self.hop = self.win_samples // self.n_time                   # 312
        n_fft = int(round(self.hop * FRAME_TO_HOP_RATIO))            # 780
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=n_fft, hop_length=self.hop, n_mels=self.n_mels,
            f_min=F_MIN, f_max=SR // 2, power=2.0,
        )
        self.n_layers = depth
        self.embed_dim = embed_dim
        self._downsample = self.hop * (self.n_time // self.grid_w)   # 312 * 16 = 4992

    def get_downsample_rates(self, key: str) -> int:
        return self._downsample

    def _logmel_windows(self, wav: Tensor) -> Tensor:
        """wav (L,) -> (n_win, 1, n_mels, n_time); windowing identical to expert.py."""
        L = wav.shape[-1]
        n_win = max(1, math.ceil(L / self.win_samples))
        pad_to = n_win * self.win_samples
        if L < pad_to:
            wav = F.pad(wav, (0, pad_to - L))
        windows = wav.reshape(n_win, self.win_samples)

        mel = torch.log(self.mel(windows) + 1e-6)
        T = mel.shape[-1]
        if T >= self.n_time:
            mel = mel[..., : self.n_time]
        else:
            mel = F.pad(mel, (0, self.n_time - T), value=float(mel.min()))
        mel = (mel - self.norm_mean) / self.norm_std
        return mel.unsqueeze(1)

    @torch.no_grad()
    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        device = next(self.encoder.parameters()).device
        per_layer: List[List[Tensor]] = [[] for _ in range(self.n_layers)]

        for wav in wavs:
            wav = wav.to(device).float()
            if wav.dim() > 1:
                wav = wav.reshape(-1)
            L = wav.shape[-1]
            x = self._logmel_windows(wav)
            n_win = x.shape[0]
            feats = self.encoder.forward_intermediate(x)
            keep = max(1, min(round(L / self._downsample), n_win * self.grid_w))

            for li, f in enumerate(feats):
                if FREQ_MAJOR:
                    f = f.view(n_win, self.grid_h, self.grid_w, self.embed_dim).mean(dim=1)
                else:
                    f = f.view(n_win, self.grid_w, self.grid_h, self.embed_dim).mean(dim=2)
                f = f.reshape(n_win * self.grid_w, self.embed_dim)[:keep]
                per_layer[li].append(f)

        hidden_states = [pad_sequence(layer, batch_first=True) for layer in per_layer]
        return {"hidden_states": hidden_states}
