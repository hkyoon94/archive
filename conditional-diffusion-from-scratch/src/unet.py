from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from yacs.config import CfgNode


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim        
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim])  # normalizes over last, embedding dimension
        self.ff = nn.Sequential(
            nn.LayerNorm([embed_dim]),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: Tensor):
        bsz, _, height, width = x.shape
        # self-attention for every pixel-embedded features
        x = x.view(bsz, self.embed_dim, height * width).swapaxes(1, 2)  # embed_dim to last
        x_ln = self.ln(x)
        x_attn, _ = self.mha(query=x_ln, key=x_ln, value=x_ln)
        x_attn = x_attn + x  # residual connection with attn block
        x_out: Tensor = self.ff(x_attn) + x_attn  # residual connection with ff
        x_out = x_out.view(bsz, height * width, self.embed_dim).swapaxes(2, 1)
        x_out = x_out.view(bsz, self.embed_dim, height, width)
        return x_out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.down = nn.MaxPool2d(kernel_size=2)
        self.t_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels),
        )
        self.self_attn = MultiHeadSelfAttention(embed_dim=out_channels)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_skip = self.conv(x)
        x = self.down(x_skip)
        t_emb = self.t_linear(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x = self.self_attn(x + t_emb)
        return x, x_skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.t_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels),
        )
        self.self_attn = MultiHeadSelfAttention(embed_dim=out_channels)

    def forward(self, x: Tensor, x_skip: Tensor, t: Tensor) -> Tensor:
        x = self.up(x)
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)
        t_emb = self.t_linear(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x = self.self_attn(x + t_emb)
        return x


class UNet(nn.Module):
    def __init__(self, cfg: CfgNode, c_in: int = 3, c_out: int = 3):
        super().__init__()
        self.t_emb_dim: int = cfg.UNET.T_EMBED_DIMENSION
        down_outs = cfg.UNET.DOWN_BLOCK_OUTS
        bottom_outs = cfg.UNET.BOTTOM_BLOCK_OUTS
        up_outs = cfg.UNET.UP_BLOCK_OUTS

        # building down-blocks
        self.down_blocks = nn.ModuleList([])
        self.down_blocks.append(
            DownBlock(c_in, down_outs[0], t_emb_dim=self.t_emb_dim)
        )
        for i in range(len(down_outs) - 1):
            self.down_blocks.append(
                DownBlock(down_outs[i], down_outs[i + 1], t_emb_dim=self.t_emb_dim)
            )

        # building bottom-blocks
        self.bottom_blocks = nn.ModuleList([])
        self.bottom_blocks.append(
            DoubleConv(down_outs[-1], bottom_outs[0])
        )
        for i in range(len(bottom_outs) - 1):
            self.bottom_blocks.append(
                DoubleConv(bottom_outs[i], bottom_outs[i + 1])
            )

        # building up-blocks
        self.up_blocks = nn.ModuleList([])
        self.up_blocks.append(
            UpBlock(bottom_outs[-1] + down_outs[-1], up_outs[0])
        )
        for i in range(len(up_outs) - 1):
            # 주의: up-block들은 skip-concat를 위해 in_channel이 pair down-block의 채널 수만큼 더 많아야 함
            self.up_blocks.append(
                UpBlock(down_outs[-2 - i] + up_outs[i], up_outs[i + 1], t_emb_dim=self.t_emb_dim)
            )

        self.out_layer = nn.Conv2d(
            in_channels=up_outs[-1], out_channels=c_out, kernel_size=1
        )

    def pos_encoding(self, t, emb_dim) -> Tensor:
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, emb_dim, 2, device=one_param(self).device).float() / emb_dim)
        )
        pos_enc_a = torch.sin(t.repeat(1, emb_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, emb_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forward(self, x: Tensor, t: Tensor) -> Tensor:
        # down-sampling
        x_skips = []
        for down_block in self.down_blocks:
            x, x_skip = down_block(x, t)  # (b x c_in x h_in x w_in) -> (b x c_out x h_in / 2 x w_in / 2)
            x_skips.insert(0, x_skip)  # caching for skip-concat

        # bottom-conv
        for bottom_block in self.bottom_blocks:
            x = bottom_block(x)

        # up-sampling
        for up_block, x_skip in zip(self.up_blocks, x_skips):
            x = up_block(x, x_skip, t)

        return self.out_layer(x)

    def forward(self, x: Tensor, t: Tensor):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.t_emb_dim)
        return self.unet_forward(x, t)


class UNetConditional(UNet):
    def __init__(
        self, cfg: CfgNode, c_in: int = 3, c_out: int = 3, num_classes: Optional[int] = None, **kwargs
    ):
        super().__init__(cfg, c_in, c_out, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, embedding_dim=cfg.UNET.T_EMBED_DIMENSION)

    def forward(self, x: Tensor, t: Tensor, y=None) -> Tensor:
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.t_emb_dim)

        if y is not None:
            """conditional embedding"""
            t = t + self.label_emb(y)

        return self.unet_forward(x, t)


def one_param(m: nn.Module):
    return next(iter(m.parameters()))
