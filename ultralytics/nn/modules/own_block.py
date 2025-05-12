import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from einops import rearrange

__all__= (
    "DepthwiseSeparableConv",
    "WaveletDownsampleWrapper",
    "CED",
)


class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=True):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels
        self.grad = grad

    def forward(self, x, rev=False):
        if not rev:
            # 正向小波变换
            out = x.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = out.permute(0, 2, 1, 3, 4).reshape([x.shape[0], 4 * self.in_channels, x.shape[2] // 2, x.shape[3] // 2])
        else:
            # 逆向小波变换
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = out.permute(0, 2, 1, 3, 4).reshape([x.shape[0], self.in_channels, x.shape[2] * 2, x.shape[3] * 2])
        return out

class WFD(nn.Module):
    def __init__(self, dim_in, dim, need=False):
        super(WFD, self).__init__()
        self.need = need
        if need:
            self.first_conv = nn.Conv2d(dim_in, dim, kernel_size=1, padding=0)
            self.HaarWavelet = HaarWavelet(dim, grad=False)
            self.dim = dim
        else:
            self.HaarWavelet = HaarWavelet(dim_in, grad=False)
            self.dim = dim_in

    def forward(self, x):
        if self.need:
            x = self.first_conv(x)
        
        haar = self.HaarWavelet(x, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim*2, self.dim) 
        d = haar.narrow(1, self.dim*3, self.dim)

        return a, h+v+d

class WaveletDownsampleWrapper(nn.Module):
    def __init__(self, in_channels, use_conv=True):
        super().__init__()
        k=3
        d=2
        self.wfd = WFD(dim_in=in_channels, dim=in_channels, need=use_conv)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, k, padding=k//2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, k, stride=1, padding=((k//2)*d), groups=in_channels, dilation=d))      
    def forward(self, x):
        a, hvd = self.wfd(x)
        B, C, H, W = a.shape
        a_flat = a.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        attn_out, _ = self.attn(a_flat, a_flat, a_flat)  # [B, H*W, C]
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        conv_out = self.conv(hvd)  # [B, C, H, W]

        return attn_out + conv_out  # [B, C, H, W]
    
def autopad(k, p=None):
    """
    Automatically set padding = k//2 if not specified, to preserve spatial dims.
    """
    return k // 2 if p is None else p


class CED(nn.Module):
    """
    Channel Expansion + Depthwise (CED) block:
      1) 1x1 conv to reduce channels to c = int(c2 * e)
      2) depthwise conv 3x3
      3) spatial downsample by splitting and concatenating 2x2 grids
      4) 1x1 conv to expand back to c2 channels

    Args:
        c1 (int): input channels
        c2 (int): output channels
        e (float): expansion ratio for intermediate channels
    """
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        # intermediate channels
        self.c = int(c2 * e)

        # reduce channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, self.c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU()
        )

        # depthwise conv
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                self.c, self.c,
                kernel_size=3, stride=1,
                padding=autopad(3),
                groups=self.c,
                bias=False
            ),
            nn.BatchNorm2d(self.c),
            nn.SiLU()
        )

        # expand + merge
        self.cv2 = nn.Sequential(
            nn.Conv2d(self.c * 4, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2)
            # no activation by default; add nn.SiLU() if desired
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1x1 -> depthwise
        x = self.dwconv(self.cv1(x))

        # split into 4 sub-samplings
        # top-left, bottom-left, top-right, bottom-right
        x = torch.cat([
            x[..., ::2, ::2],    # even rows, even cols
            x[..., 1::2, ::2],   # odd rows, even cols
            x[..., ::2, 1::2],   # even rows, odd cols
            x[..., 1::2, 1::2],  # odd rows, odd cols
        ], dim=1)

        # final 1x1 conv to c2
        return self.cv2(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bottleneck_ratio=0.5, groups=4):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步幅
        :param padding: 填充
        :param bottleneck_ratio: 用于瓶颈层的通道缩减比例
        :param groups: 逐点卷积的分组数
        """
        super().__init__()
        bottleneck_channels = int(in_channels * bottleneck_ratio)
        
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels
        )
        
        # 瓶颈层降维
        self.bottleneck = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0)
        
        # 分组逐点卷积
        self.pointwise = nn.Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bottleneck(x)
        x = self.pointwise(x)
        return x
