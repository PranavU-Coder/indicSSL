from __future__ import annotations
import torch
import torch.nn as nn

from src.indicJEPA.ajepa import ViTEncoder, Block, get_2d_sincos_pos_embed, _init_weights


class MAEDecoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, depth, num_heads,
                 grid_h, grid_w, patch_pixels, mlp_ratio=4.0):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.n_patches = grid_h * grid_w

        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)          
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))    
        pos = get_2d_sincos_pos_embed(decoder_dim, grid_h, grid_w)
        self.register_buffer("pos_embed", pos.unsqueeze(0))            
        self.blocks = nn.ModuleList(                                     
            [Block(decoder_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(decoder_dim)                            
        self.pred = nn.Linear(decoder_dim, patch_pixels)                
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, x, keep_indices, mask_indices):
        """
        x:            (B, K, encoder_dim)   encoded visible patches
        keep_indices: (B, K)                indices of visible patches
        mask_indices: (B, M)                indices of masked patches
        returns:      (B, M, patch_pixels)  predicted pixels at masked positions
        """
        B = x.size(0)
        x = self.decoder_embed(x)                                        # (B, K, Dd)
        Dd = x.size(-1)

        full = self.mask_token.to(x.dtype).expand(B, self.n_patches, Dd).clone()     # (B, N, Dd)
        full.scatter_(1, keep_indices.unsqueeze(-1).expand(-1, -1, Dd), x)

        full = full + self.pos_embed                                    
        for blk in self.blocks:
            full = blk(full)
        full = self.norm(full)
        pred = self.pred(full)                                           # (B, N, patch_pixels)

        # return predictions at masked positions only
        idx = mask_indices.unsqueeze(-1).expand(-1, -1, pred.size(-1))
        return torch.gather(pred, 1, idx)                                # (B, M, patch_pixels)


class IndicMAE(nn.Module):
    def __init__(self, n_mels, n_frames, patch_size, embed_dim, depth, num_heads,
                 decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = ViTEncoder(n_mels, n_frames, patch_size, embed_dim,
                                  depth, num_heads, mlp_ratio)            
        self.grid_h = self.encoder.grid_h
        self.grid_w = self.encoder.grid_w
        self.n_patches = self.encoder.n_patches
        patch_pixels = patch_size * patch_size                           # 256
        self.decoder = MAEDecoder(embed_dim, decoder_embed_dim, decoder_depth,
                                  decoder_num_heads, self.grid_h, self.grid_w,
                                  patch_pixels, mlp_ratio)               

    def patchify(self, imgs):
        """(B,1,H,W) -> (B, N, patch_size*patch_size)"""
        p = self.patch_size
        B, _, H, W = imgs.shape
        gh, gw = H // p, W // p
        x = imgs.reshape(B, 1, gh, p, gw, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, gh * gw, p * p)
        return x                                                         # (B, N, p*p)

    @staticmethod
    def random_masking(B, n_patches, mask_ratio, device):
        n_keep = int(round(n_patches * (1.0 - mask_ratio)))
        noise = torch.rand(B, n_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        keep = ids_shuffle[:, :n_keep]
        mask = ids_shuffle[:, n_keep:]
        return keep, mask

    def forward(self, imgs, mask_ratio_min=0.4, mask_ratio_max=0.6, norm_pix_loss=True):
        """imgs: (B,1,n_mels,n_frames). Returns scalar MSE loss on masked patches."""
        B = imgs.size(0)
        device = imgs.device
        mask_ratio = float(torch.empty(1).uniform_(mask_ratio_min, mask_ratio_max).item())

        keep_idx, mask_idx = self.random_masking(B, self.n_patches, mask_ratio, device)

        latent = self.encoder(imgs, keep_indices=keep_idx)               # (B, K, D)
        pred = self.decoder(latent, keep_idx, mask_idx)                  # (B, M, p*p)

        target_all = self.patchify(imgs)                                 # (B, N, p*p)
        target = torch.gather(
            target_all, 1, mask_idx.unsqueeze(-1).expand(-1, -1, target_all.size(-1))
        )                                                                # (B, M, p*p)

        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = ((pred - target) ** 2).mean()
        return loss


def build_from_config(cfg):
    """Instantiate"""
    m = cfg["model"]
    d = cfg["data"]
    return IndicMAE(
        n_mels=int(d["n_mels"]),
        n_frames=int(d["n_time_bins"]),
        patch_size=int(m["patch_size"]),
        embed_dim=int(m["embed_dim"]),
        depth=int(m["depth"]),
        num_heads=int(m["num_heads"]),
        decoder_embed_dim=int(m["predictor_embed_dim"]),      # 192 (config reused JEPA field names)
        decoder_depth=int(m["predictor_depth"]),              # 6
        decoder_num_heads=int(m["predictor_num_heads"]),      # 6
        mlp_ratio=float(m.get("mlp_ratio", 4.0)),
    )