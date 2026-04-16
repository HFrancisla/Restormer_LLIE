## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from pdb import set_trace as stx
import numbers

from einops import rearrange
from .extra_attention_raw import HTA, WTA
from .haar_wavelets import DWT_2D, IDWT_2D


##########################################################################
## Layer Norm


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        attn_type="MDTA",
        use_checkpoint=False,
    ):
        super(TransformerBlock, self).__init__()
        self.use_checkpoint = use_checkpoint

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if attn_type == "MDTA":
            self.attn = Attention(dim, num_heads, bias)
        elif attn_type == "HTA":
            self.attn = HTA(dim, num_heads, bias)
        elif attn_type == "WTA":
            self.attn = WTA(dim, num_heads, bias)
        # elif (
        #     attn_type == "IRS"
        # ):  # Intra-Row Self-Attention, which is vertical attention
        #     self.attn = IRS(dim, num_heads, bias)
        # elif (
        #     attn_type == "ICS"
        # ):  # Intra-Column Self-Attention, which is horizontal attention
        #     self.attn = ICS(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        # 3×3 Conv 先将通道数减半：C → C/2
        # PixelUnshuffle(2) 将空间尺寸缩小 2×，同时通道数 ×4：C/2 → C/2×4 = 2C
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class DWT_Downsample(nn.Module):
    """DWT-based downsampling: LL as downsampled features, LH/HL/HH as detail skip."""

    def __init__(self, wave="haar"):
        super(DWT_Downsample, self).__init__()
        self.dwt = DWT_2D(wave)

    def forward(self, x):
        # x: (B, C, H, W)
        dwt_out = self.dwt(x)  # (B, 4C, H/2, W/2)
        C = dwt_out.shape[1] // 4
        x_ll = dwt_out[:, :C, :, :]  # (B, C, H/2, W/2)
        x_detail = dwt_out[:, C:, :, :]  # (B, 3C, H/2, W/2) = [LH, HL, HH]
        return x_ll, x_detail


class IDWT_Upsample(nn.Module):
    """IDWT-based upsampling: combines decoder features (as LL) with encoder detail coefficients."""

    def __init__(self, wave="haar"):
        super(IDWT_Upsample, self).__init__()
        self.idwt = IDWT_2D(wave)

    def forward(self, x, detail):
        # x: (B, C, H/2, W/2) - decoder features (LL)
        # detail: (B, 3C, H/2, W/2) - [LH, HL, HH] from encoder DWT
        idwt_in = torch.cat([x, detail], dim=1)  # (B, 4C, H/2, W/2)
        out = self.idwt(idwt_in)  # (B, C, 2H, 2W)
        return out


##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
        attn_types=["MDTA", "MDTA", "MDTA", "MDTA"],
        dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        use_checkpoint=False,
    ):
        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[0],
                    use_checkpoint=use_checkpoint,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = DWT_Downsample()  ## From Level 1 to Level 2 (DWT)
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[1],
                    use_checkpoint=use_checkpoint,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = DWT_Downsample()  ## From Level 2 to Level 3 (DWT)
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[2],
                    use_checkpoint=use_checkpoint,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = DWT_Downsample()  ## From Level 3 to Level 4 (DWT)
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[3],
                    use_checkpoint=use_checkpoint,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = IDWT_Upsample()  ## From Level 4 to Level 3 (IDWT)
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2), dim, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[2],
                    use_checkpoint=use_checkpoint,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = IDWT_Upsample()  ## From Level 3 to Level 2 (IDWT)
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2), dim, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[1],
                    use_checkpoint=use_checkpoint,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = IDWT_Upsample()  ## From Level 2 to Level 1 (IDWT)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[0],
                    use_checkpoint=use_checkpoint,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    attn_type=attn_types[0],
                    use_checkpoint=use_checkpoint,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2, detail1 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3, detail2 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4, detail3 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent, detail3)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3, detail2)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2, detail1)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
