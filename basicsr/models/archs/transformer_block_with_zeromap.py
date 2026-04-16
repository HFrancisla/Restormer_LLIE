import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


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


class Inter_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Inter_Attention, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type="WithBias")
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

    def forward(self, x, zero_map):
        b, c, h, w = x.shape
        m = x
        x = self.norm1(x)

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(2, 3)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)

        V1 = out + m

        return V1


class Intra_Attention_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias, N=8):
        super(Intra_Attention_withZeromap, self).__init__()

        self.N = N
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type="WithBias")
        self.qkv = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 4,
            dim * 4,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 4,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, zero_map):
        b, c, h, w = x.shape
        m = x
        x = self.norm1(x)
        h_pad = self.N - h % self.N if not h % self.N == 0 else 0
        w_pad = self.N - w % self.N if not w % self.N == 0 else 0
        x = F.pad(x, (0, w_pad, 0, h_pad), "reflect")
        zero_map = F.pad(zero_map, (0, w_pad, 0, h_pad), "reflect")
        zero_map = F.interpolate(zero_map, (x.shape[2], x.shape[3]), mode="bilinear")
        zero_map[zero_map <= 0.2] = 0
        zero_map[zero_map > 0.2] = 1
        b, c, H, W = x.shape
        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v, v1 = qkv.chunk(4, dim=1)

        q = rearrange(
            q,
            "b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)",
            head=self.num_heads,
            N1=self.N,
            N2=self.N,
        )
        k = rearrange(
            k,
            "b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)",
            head=self.num_heads,
            N1=self.N,
            N2=self.N,
        )
        v = rearrange(
            v,
            "b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)",
            head=self.num_heads,
            N1=self.N,
            N2=self.N,
        )
        v1 = rearrange(
            v1,
            "b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)",
            head=self.num_heads,
            N1=self.N,
            N2=self.N,
        )
        q_zero = rearrange(
            zero_map,
            "b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)",
            head=self.num_heads,
            N1=self.N,
            N2=self.N,
        )
        k_zero = rearrange(
            zero_map,
            "b (head c) (h1 N1)  (w1 N2) -> b head c (N1 N2) (h1 w1)",
            head=self.num_heads,
            N1=self.N,
            N2=self.N,
        )

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(3, 4)) * self.temperature1
        attn = attn.softmax(dim=-1)
        out = attn @ v

        attn_zero = (q_zero @ k_zero.transpose(3, 4)) * self.temperature2
        attn_zero = attn_zero.softmax(dim=-1)
        out_zero = attn_zero @ v1

        out = rearrange(
            out,
            "b head c (N1 N2) (h1 w1) -> b (head c) (h1 N1)  (w1 N2)",
            head=self.num_heads,
            N1=self.N,
            N2=self.N,
            h1=H // self.N,
            w1=W // self.N,
        )
        out_zero = rearrange(
            out_zero,
            "b head c (N1 N2) (h1 w1) -> b (head c) (h1 N1)  (w1 N2)",
            head=self.num_heads,
            N1=self.N,
            N2=self.N,
            h1=H // self.N,
            w1=W // self.N,
        )

        out = self.project_out(out)
        out_zero = self.project_out(out_zero)

        out = out[:, :, :h, :w]
        out_zero = out_zero[:, :, :h, :w]

        V1 = out + m
        V2 = out_zero + m

        return V1 + V2


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


class IIZAT(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor=2.66,
        bias=True,
        LayerNorm_type="WithBias",
        N=4,
    ):
        super(IIZAT, self).__init__()

        self.attn_inter = Inter_Attention(dim, num_heads, bias)
        self.attn_intra = Intra_Attention_withZeromap(dim, num_heads, bias, N=N)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, zero_map):
        m = self.attn_inter(x, zero_map)
        z = self.attn_intra(m, zero_map)

        out = z + self.ffn(self.norm2(z))

        return out
