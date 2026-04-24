"""
可视化脚本公共模块 (不可直接运行)
========================================
功能:
    为 visualize_attention.py 和 visualize_first_block.py 提供共享组件:
    - 网络层: LayerNorm, FeedForward, OverlapPatchEmbed, Downsample
    - 注意力模块: AttentionWithOutputs (MDTA/HTA/WTA), TransformerBlockWithMap
    - 可视化辅助: save_figure, save_figure_no_colorbar, compute_l2_norm
    - Checkpoint 工具: load_checkpoint, load_module_from_state_dict, attn_v_to_4d

使用方法:
    本模块为库文件，由其他可视化脚本导入，不可直接运行。
    from common import load_image, save_figure, TransformerBlockWithMap, ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from PIL import Image
from torchvision import transforms
import numbers
import os


# ============== Layer Norm ==============


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ============== Feed Forward & Patch Embed & Downsample ==============


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
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
        return self.project_out(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ============== 统一注意力模块 ==============

# rearrange 配置: attn_type -> (forward_pattern, inverse_pattern, extra_dims_key)
REARRANGE_CONFIG = {
    "MDTA": {
        "forward": "b (head c) h w -> b head c (h w)",
        "inverse": "b head c (h w) -> b (head c) h w",
        "extra_dims": lambda h, w: {"h": h, "w": w},
    },
    "HTA": {
        "forward": "b (head c) h w -> b head w (c h)",
        "inverse": "b head w (c h) -> b (head c) h w",
        "extra_dims": lambda h, w: {"h": h, "w": w},
    },
    "WTA": {
        "forward": "b (head c) h w -> b head h (c w)",
        "inverse": "b head h (c w) -> b (head c) h w",
        "extra_dims": lambda h, w: {"h": h, "w": w},
    },
}


class AttentionWithOutputs(nn.Module):
    """统一的注意力模块，支持 MDTA/HTA/WTA，可保存所有中间结果"""

    def __init__(
        self, dim, num_heads, bias, attn_type="MDTA", save_intermediates=False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.save_intermediates = save_intermediates
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

        self._cfg = REARRANGE_CONFIG[attn_type]

        # 基本输出（始终保存）
        self.attn_map = None
        self.value_map = None
        self.attn_v_map = None

        # 详细中间结果（仅 save_intermediates=True 时保存）
        self.q_before_rearrange = None
        self.k_before_rearrange = None
        self.v_before_rearrange = None
        self.q_after_rearrange = None
        self.k_after_rearrange = None
        self.v_after_rearrange = None
        self.attn_v_before_rearrange = None
        self.attn_v_after_rearrange = None

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        if self.save_intermediates:
            self.q_before_rearrange = q.clone().detach()
            self.k_before_rearrange = k.clone().detach()
            self.v_before_rearrange = v.clone().detach()

        fwd = self._cfg["forward"]
        inv = self._cfg["inverse"]
        extra = self._cfg["extra_dims"](h, w)

        q = rearrange(q, fwd, head=self.num_heads)
        k = rearrange(k, fwd, head=self.num_heads)
        v = rearrange(v, fwd, head=self.num_heads)

        if self.save_intermediates:
            self.q_after_rearrange = q.clone().detach()
            self.k_after_rearrange = k.clone().detach()
            self.v_after_rearrange = v.clone().detach()

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        self.attn_map = attn.detach()
        self.value_map = v.detach()

        out = attn @ v
        self.attn_v_map = out.detach()

        if self.save_intermediates:
            self.attn_v_before_rearrange = out.detach()

        out = rearrange(out, inv, head=self.num_heads, **extra)

        if self.save_intermediates:
            self.attn_v_after_rearrange = out.detach()

        return self.project_out(out)


# ============== TransformerBlock ==============


class TransformerBlockWithMap(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        attn_type="MDTA",
        save_intermediates=False,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttentionWithOutputs(
            dim, num_heads, bias, attn_type, save_intermediates
        )
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.attn_output = None

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        self.attn_output = attn_out.detach()
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

    def get_attn_map(self):
        return self.attn.attn_map

    def get_attn_output(self):
        return self.attn_output

    def get_value_map(self):
        return self.attn.value_map

    def get_attn_v_map(self):
        return self.attn.attn_v_map


# ============== Utility Functions ==============


def load_image(image_path, size=256, no_resize=True):
    img = Image.open(image_path).convert("RGB")
    if no_resize:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ]
        )
    return transform(img).unsqueeze(0)


def compute_l2_norm(tensor):
    """计算 L2 Norm: sqrt(sum(x^2)) across channel dimension"""
    if tensor.dim() == 4:  # [B, C, H, W]
        return torch.sqrt((tensor[0] ** 2).sum(dim=0)).cpu().numpy()
    elif tensor.dim() == 3:  # [C, H, W]
        return torch.sqrt((tensor**2).sum(dim=0)).cpu().numpy()
    else:
        return tensor.cpu().numpy()


def get_level_name(level):
    """获取 level 显示名称"""
    return "Latent" if level == 4 else f"Encoder L{level}"


# ============== 可视化辅助函数 ==============


def save_figure(
    data,
    save_dir,
    filename,
    simple_dir_replace=None,
    title=None,
    suptitle=None,
    cmap="viridis",
    aspect=None,
    xlabel=None,
    ylabel=None,
    figsize=(8, 6),
    dpi=150,
):
    """统一的图片保存函数，同时生成完整版和简化版（用于论文）

    Args:
        data: 2D numpy array
        save_dir: 保存目录
        filename: 文件名
        simple_dir_replace: (old, new) tuple，用于生成简化版目录路径
        title: 子图标题
        suptitle: 总标题
        cmap: colormap
        aspect: imshow aspect 参数
        xlabel, ylabel: 轴标签
        figsize: 图片大小
        dpi: 分辨率
    """
    os.makedirs(save_dir, exist_ok=True)

    # 完整版本（带标题和标签）
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect=aspect)
    if title:
        ax.set_title(title, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if not xlabel and not ylabel:
        ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if suptitle:
        plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    # 简化版本（去掉所有标题和标签，用于论文）
    if simple_dir_replace:
        save_dir_simple = save_dir.replace(simple_dir_replace[0], simple_dir_replace[1])
        os.makedirs(save_dir_simple, exist_ok=True)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data, cmap=cmap, aspect=aspect)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        save_path_simple = os.path.join(save_dir_simple, filename)
        plt.savefig(save_path_simple, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved (simple): {save_path_simple}")


def save_figure_no_colorbar(
    data,
    save_dir,
    filename,
    simple_dir_replace=None,
    cmap="viridis",
    aspect=None,
    figsize=(10, 8),
    dpi=200,
):
    """保存不带 colorbar 的简洁图片（完整+简化版）"""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect=aspect)
    ax.axis("off")
    plt.tight_layout(pad=0)

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved: {save_path}")

    if simple_dir_replace:
        save_dir_simple = save_dir.replace(simple_dir_replace[0], simple_dir_replace[1])
        os.makedirs(save_dir_simple, exist_ok=True)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data, cmap=cmap, aspect=aspect)
        ax.axis("off")
        plt.tight_layout(pad=0)

        save_path_simple = os.path.join(save_dir_simple, filename)
        plt.savefig(save_path_simple, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved (simple): {save_path_simple}")


# ============== Checkpoint 加载工具 ==============


def load_checkpoint(checkpoint_path, device):
    """加载 Restormer checkpoint 并返回 state_dict"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "params" in state_dict:
        state_dict = state_dict["params"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    return state_dict


def load_module_from_state_dict(state_dict, prefix, module, name):
    """从 state_dict 中按前缀加载模块权重"""
    module_dict = {
        k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)
    }
    if module_dict:
        module.load_state_dict(module_dict, strict=False)
        print(f"Loaded: {name}")


def attn_v_to_4d(attn_type, attn_v_map, block_input, num_heads):
    """将 attention*V 从注意力空间转回 [B, C, H, W] 格式"""
    h, w = block_input.shape[2], block_input.shape[3]

    if attn_type == "MDTA":
        return rearrange(
            attn_v_map, "b head c (h w) -> b (head c) h w", head=num_heads, h=h, w=w
        )
    elif attn_type == "HTA":
        b, head, w_dim, ch = attn_v_map.shape
        c_per_head = ch // h
        attn_v_reshaped = attn_v_map.reshape(b, head, w_dim, c_per_head, h)
        return rearrange(attn_v_reshaped, "b head w c h -> b (head c) h w")
    else:  # WTA
        b, head, h_dim, cw = attn_v_map.shape
        c_per_head = cw // w
        attn_v_reshaped = attn_v_map.reshape(b, head, h_dim, c_per_head, w)
        return rearrange(attn_v_reshaped, "b head h c w -> b (head c) h w")
