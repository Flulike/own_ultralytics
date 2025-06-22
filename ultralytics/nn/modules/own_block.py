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
    "GatedCNNBlock",
    "GatedC3k2",
    "GatedCNNBlockWrapper",
    "GatedBottleneck",
    "GatedC3k",
    "GatedFFN",
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
    

class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut


class GatedC3k2(nn.Module):
    """
    GatedC3k2: A CSP-style module using GatedCNNBlock that can replace C3k2
    完全兼容tasks.py中的参数传递机制
    """
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, g=1, **kwargs):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([
            GatedCNNBlockWrapper(self.c) for _ in range(n)
        ])
        self.shortcut = shortcut

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class GatedCNNBlockWrapper(nn.Module):
    """
    Wrapper for GatedCNNBlock to handle tensor format conversion
    and make it compatible with C3k2 interface.
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0, shortcut=True):
        super().__init__()
        self.gated_block = GatedCNNBlock(
            dim=dim,
            expansion_ratio=expansion_ratio,
            kernel_size=kernel_size,
            conv_ratio=conv_ratio
        )
        self.shortcut = shortcut
        
    def forward(self, x):
        """
        Forward pass with tensor format conversion.
        Input: [B, C, H, W] -> Output: [B, C, H, W]
        """
        # Convert from [B, C, H, W] to [B, H, W, C] for GatedCNNBlock
        x_permuted = x.permute(0, 2, 3, 1)
        
        # Apply GatedCNNBlock
        out_permuted = self.gated_block(x_permuted)
        
        # Convert back to [B, C, H, W]
        out = out_permuted.permute(0, 3, 1, 2)
        
        # Add shortcut if enabled and dimensions match
        if self.shortcut:
            return out + x
        return out

class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network (GatedFFN) module.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels  
        n (int): Number of repetitions
        shortcut (bool): Whether to use shortcut connection
        g (int): Groups for convolution
        e (float): Expansion ratio for hidden channels
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, 
                 g: int = 1, e: float = 0.5):
        super().__init__()
        self.n = n
        self.c = int(c2 * e)  # hidden channels
        
        # Projection layer: 1x1 conv to split into 2 parts
        self.proj = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU()
        )
        
        # RepDWConv equivalent using RepConv
        self.rep = RepConv(self.c, self.c, k=3, s=1, p=1, g=self.c, act=True)
        
        # Additional depthwise conv layers if n > 1
        self.m = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.c, self.c, kernel_size=3, stride=1, padding=1, groups=self.c, bias=False),
                nn.BatchNorm2d(self.c)
            ) for _ in range(n - 1)
        ])
        
        # GELU activation
        self.act = nn.GELU()
        
        # Final projection layer
        self.cv2 = nn.Sequential(
            nn.Conv2d(self.c, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2)
        )
        
        # Shortcut connection
        self.add = shortcut and c1 == c2

    def forward(self, x):
        shortcut = x.clone()
        
        # Split projection into two parts
        x, z = self.proj(x).split([self.c, self.c], 1)
        
        # Apply RepConv
        x = self.rep(x)
        
        # Apply additional depthwise convs if n > 1
        if self.n != 1:
            for m in self.m:
                x = m(x)
        
        # Gated mechanism: multiply with activated z
        x = x * self.act(z)
        
        # Final projection
        x = self.cv2(x)
        
        # Add shortcut if enabled
        return x + shortcut if self.add else x