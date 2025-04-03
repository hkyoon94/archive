import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
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


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, t_emb_dim=256):
        super().__init__()
        self.t_emb_dim = t_emb_dim

        self.inc = DoubleConv(in_channels=c_in, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128, t_emb_dim=t_emb_dim)
        self.sa1 = SelfAttention(embed_dim=128)
        self.down2 = Down(in_channels=128, out_channels=256, t_emb_dim=t_emb_dim)
        self.sa2 = SelfAttention(embed_dim=256)
        self.down3 = Down(in_channels=256, out_channels=256, t_emb_dim=t_emb_dim)
        self.sa3 = SelfAttention(embed_dim=256)

        self.bot1 = DoubleConv(in_channels=256, out_channels=512)
        self.bot2 = DoubleConv(in_channels=512, out_channels=512)
        self.bot3 = DoubleConv(in_channels=512, out_channels=256)

        self.up1 = Up(in_channels=512, out_channels=128, t_emb_dim=t_emb_dim)
        self.sa4 = SelfAttention(embed_dim=128)
        self.up2 = Up(in_channels=256, out_channels=64, t_emb_dim=t_emb_dim)
        self.sa5 = SelfAttention(embed_dim=64)
        self.up3 = Up(in_channels=128, out_channels=64, t_emb_dim=t_emb_dim)
        self.sa6 = SelfAttention(embed_dim=64)
        self.outc = nn.Conv2d(in_channels=64, out_channels=c_out, kernel_size=1)

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
        x1 = self.inc(x)  # (b x c_in x h x w) -> (b x c_out x h x w)
        x2 = self.down1(x1, t)  # (b x c_in x h_in x w_in) -> (b x c_out x h_out x w_out)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        return self.outc(x)

    def forward(self, x: Tensor, t: Tensor):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.t_emb_dim)
        return self.unet_forward(x, t)


class UNetConditional(UNet):
    def __init__(self, c_in=3, c_out=3, t_emb_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, t_emb_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, t_emb_dim)

    def forward(self, x: Tensor, t: Tensor, y=None) -> Tensor:
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.t_emb_dim)

        if y is not None:
            """conditional embedding"""
            t = t + self.label_emb(y)

        return self.unet_forward(x, t)


def one_param(m: nn.Module):
    return next(iter(m.parameters()))
