import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

import einops
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_

# from basicsr.archs.arch_util import LayerNorm2d
from timm.models.layers import LayerNorm2d
# from natten.functional import na2d_qk, na2d_av

__all__= (
    "DepthwiseSeparableConv",
    "WaveletDownsampleWrapper",
    "CED",
    # "GGmix",
    # "DeformableNeighborhoodAttention",
)

#region WaveletDownsampleWrapper
class HaarWavelet(nn.Module):
    # 使用例：- [-1, 1, WaveletDownsampleWrapper, [256]]
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
#endregion

class CED(nn.Module):
    # 使用例：- [-1, 1, CED,  [256, 0.5]] 
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

FUSED = True
class DeformableNeighborhoodAttention(nn.Module):

# 使用例：- [-1, 1, DeformableNeighborhoodAttention, [512, 8, 7]]  # 参数: dim, num_heads, kernel_size
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        offset_range_factor=1.0,
        stride=1,
        use_pe=True,
        dwc_pe=True,
        no_off=False,
        fixed_pe=False,
        is_causal: bool = False,
        rel_pos_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):

        super().__init__()
        n_head_channels = dim // num_heads
        n_groups = num_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = num_heads
        self.nc = n_head_channels * num_heads
        self.n_groups = num_heads
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = kernel_size
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels,
                      kk, stride, pad_size, groups=self.n_group_channels),
            LayerNorm2d(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2 * self.kernel_size[0] - 1),
                    (2 * self.kernel_size[1] - 1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.rpe_table = nn.Conv2d(
            self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key,
                           dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key,
                           dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(
            B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(
            B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(
            q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg

        Hk, Wk = offset.size(2), offset.size(3)

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor(
                [1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(
                x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(
                3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups,
                                self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, H, W)

        residual_lepe = self.rpe_table(q)

        if self.rpb is not None or not FUSED:
            q = einops.rearrange(q, 'b (g c) h w -> b g h w c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            k = einops.rearrange(self.proj_k(x_sampled), 'b (g c) h w -> b g h w c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            v = einops.rearrange(self.proj_v(x_sampled), 'b (g c) h w -> b g h w c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)

            q = q*self.scale
            attn = na2d_qk(
                q,
                k,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = na2d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            out = einops.rearrange(out, 'b g h w c -> b (g c) h w')

        else:
            q = einops.rearrange(q, 'b (g c) h w -> b h w g c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            k = einops.rearrange(self.proj_k(x_sampled), 'b (g c) h w -> b h w g c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            v = einops.rearrange(self.proj_v(x_sampled), 'b (g c) h w -> b h w g c',
                                 g=self.n_groups, b=B, c=self.n_group_channels, h=H, w=W)
            out = na2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
                scale=self.scale,
            )
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe

        y = self.proj_drop(self.proj_out(out))

        return y
#endregion