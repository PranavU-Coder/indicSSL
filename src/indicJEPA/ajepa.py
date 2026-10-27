import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class MelDataset(Dataset):
    """loads cached log-mel specs, random time-crops to n_frames, normalizes."""

    def __init__(self, cache_dir, split="dev", n_frames=256):
        self.files = sorted(Path(cache_dir, split).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no .pt files in {cache_dir}/{split}.")
        self.n_frames = n_frames
        print(f"[MelDataset {split}] {len(self.files)} cached specs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = torch.load(self.files[idx], weights_only=True).float()
        _, T = mel.shape
        if T >= self.n_frames:
            s = random.randint(0, T - self.n_frames)
            mel = mel[:, s : s + self.n_frames]
        else:
            mel = F.pad(mel, (0, self.n_frames - T), value=float(mel.min()))
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel.unsqueeze(0)


def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w):
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"
    gy = torch.arange(grid_h, dtype=torch.float32)
    gx = torch.arange(grid_w, dtype=torch.float32)
    gy, gx = torch.meshgrid(gy, gx, indexing="ij")
    eh = _sincos_1d(embed_dim // 2, gy.reshape(-1))
    ew = _sincos_1d(embed_dim // 2, gx.reshape(-1))
    return torch.cat([eh, ew], dim=1)


def _sincos_1d(dim, pos):
    assert dim % 2 == 0
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2)
    omega = 1.0 / (10000.0**omega)
    out = torch.outer(pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h),
            nn.GELU(),
            nn.Linear(h, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ViTEncoder(nn.Module):
    def __init__(
        self, n_mels, n_frames, patch_size, embed_dim, depth, num_heads, mlp_ratio=4.0
    ):
        super().__init__()
        assert n_mels % patch_size == 0, (
            f"n_mels ({n_mels}) must be divisible by patch_size ({patch_size})"
        )
        assert n_frames % patch_size == 0, (
            f"n_frames ({n_frames}) must be divisible by patch_size ({patch_size})"
        )
        self.grid_h = n_mels // patch_size
        self.grid_w = n_frames // patch_size
        self.n_patches = self.grid_h * self.grid_w
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(1, embed_dim, patch_size, patch_size)
        pos = get_2d_sincos_pos_embed(embed_dim, self.grid_h, self.grid_w)
        self.register_buffer("pos_embed", pos.unsqueeze(0))

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_init_weights)

    def forward(self, x, keep_indices=None):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        if keep_indices is not None:
            D = x.size(-1)
            idx = keep_indices.unsqueeze(-1).expand(-1, -1, D)
            x = torch.gather(x, 1, idx)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_intermediate(self, x, keep_indices=None):
        """Per-block outputs (list of (B, N, D)), taken BEFORE the final self.norm.

        Byte-for-byte identical to forward() up to and including the block loop, but
        collects every block's output instead of returning only norm(last). Used to build
        data2vec / WavJEPA-style multi-layer targets (average of the top-K teacher layers).
        """
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        if keep_indices is not None:
            D = x.size(-1)
            idx = keep_indices.unsqueeze(-1).expand(-1, -1, D)
            x = torch.gather(x, 1, idx)
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats


class Predictor(nn.Module):
    def __init__(
        self,
        encoder_dim,
        predictor_dim,
        grid_h,
        grid_w,
        depth,
        num_heads,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.n_patches = grid_h * grid_w
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        pos = get_2d_sincos_pos_embed(predictor_dim, grid_h, grid_w)
        self.register_buffer("pos_embed", pos.unsqueeze(0))

        self.blocks = nn.ModuleList(
            [Block(predictor_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, context_emb, context_idx, target_idx):
        B = context_emb.size(0)
        Dp = self.input_proj.out_features
        pos = self.pos_embed.expand(B, -1, -1)

        ctx = self.input_proj(context_emb)
        ctx = ctx + torch.gather(pos, 1, context_idx.unsqueeze(-1).expand(-1, -1, Dp))

        tgt = self.mask_token.expand(B, target_idx.size(1), -1)
        tgt = tgt + torch.gather(pos, 1, target_idx.unsqueeze(-1).expand(-1, -1, Dp))

        x = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, ctx.size(1) :])
        return self.output_proj(x)


def random_mask(batch_size, n_patches, mask_ratio_range=(0.4, 0.6), device="cpu"):
    """one mask ratio per batch (per paper Sec IV.A), different positions per sample.
    I do want to try masking ratios ranging from 0.75 - 0.8 as those were effective for MAE & M2D"""
    mr = float(torch.empty(1).uniform_(*mask_ratio_range))
    n_mask = int(round(mr * n_patches))
    n_keep = n_patches - n_mask
    noise = torch.rand(batch_size, n_patches, device=device)
    order = torch.argsort(noise, dim=1)
    return order[:, :n_keep].contiguous(), order[:, n_keep:].contiguous()


class AudioJEPA(nn.Module):
    def __init__(
        self,
        n_mels,
        n_frames,
        patch_size,
        encoder_dim,
        encoder_depth,
        encoder_heads,
        predictor_dim,
        predictor_depth,
        predictor_heads,
        mlp_ratio=4.0,
        mask_ratio_range=(0.4, 0.6),
        target_layers=1,
    ):
        super().__init__()
        self.mask_ratio_range = mask_ratio_range
        self.target_layers = int(target_layers)

        self.context_encoder = ViTEncoder(
            n_mels,
            n_frames,
            patch_size,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
        )
        self.target_encoder = ViTEncoder(
            n_mels,
            n_frames,
            patch_size,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
        )
        self._init_target_from_context()

        self.predictor = Predictor(
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            grid_h=self.context_encoder.grid_h,
            grid_w=self.context_encoder.grid_w,
            depth=predictor_depth,
            num_heads=predictor_heads,
            mlp_ratio=mlp_ratio,
        )

        self.n_patches = self.context_encoder.n_patches

    @classmethod
    def from_config(cls, cfg):
        """construct an AudioJEPA from a config dict"""
        d, m, mk = cfg["data"], cfg["model"], cfg["masking"]
        j = cfg.get("jepa", {})
        return cls(
            n_mels=d["n_mels"],
            n_frames=d["n_time_bins"],
            patch_size=m["patch_size"],
            encoder_dim=m["embed_dim"],
            encoder_depth=m["depth"],
            encoder_heads=m["num_heads"],
            predictor_dim=m["predictor_embed_dim"],
            predictor_depth=m["predictor_depth"],
            predictor_heads=m["predictor_num_heads"],
            mlp_ratio=m.get("mlp_ratio", 4.0),
            mask_ratio_range=(mk["mask_ratio_min"], mk["mask_ratio_max"]),
            target_layers=j.get("target_layers", 1),
        )

    @torch.no_grad()
    def _init_target_from_context(self):
        for p_t, p_s in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, tau):
        for p_t, p_s in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            p_t.data.mul_(tau).add_(p_s.data, alpha=1.0 - tau)

    def trainable_parameters(self):
        return list(self.context_encoder.parameters()) + list(
            self.predictor.parameters()
        )

    def _multilayer_target(self, x):
        feats = self.target_encoder.forward_intermediate(x)
        K = min(self.target_layers, len(feats))
        feats = feats[-K:]
        feats = [F.instance_norm(f.transpose(1, 2)).transpose(1, 2) for f in feats]
        tgt = torch.stack(feats, dim=0).mean(dim=0)
        tgt = F.layer_norm(tgt, (tgt.size(-1),))
        return tgt

    def forward(self, x):
        """returns (loss, ctx_emb). ctx_emb is only used for monitoring."""
        B = x.size(0)
        keep_idx, mask_idx = random_mask(
            B, self.n_patches, self.mask_ratio_range, device=x.device
        )

        ctx_emb = self.context_encoder(x, keep_indices=keep_idx)
        pred = self.predictor(ctx_emb, keep_idx, mask_idx)

        with torch.no_grad():
            if self.target_layers <= 1:
                tgt_full = self.target_encoder(x)            
            else:
                tgt_full = self._multilayer_target(x)
            D = tgt_full.size(-1)
            tgt = torch.gather(tgt_full, 1, mask_idx.unsqueeze(-1).expand(-1, -1, D))
            if self.target_layers > 1:
                tgt = tgt.to(pred.dtype)                     

        loss = F.mse_loss(pred, tgt)
        return loss, ctx_emb