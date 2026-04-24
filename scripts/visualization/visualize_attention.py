"""
Restormer Encoder 注意力机制可视化 (任意层/块)
========================================
功能:
    对 Restormer Encoder 的任意 Level (1-4) 任意 Block 进行注意力可视化，
    生成: Attention Map, Value Map, Attention*V, Feature Map, Block Output 等。
    同时输出带标注版 (完整) 和无标注版 (论文用)。

使用方法:
    python scripts/visualization/visualize_attention.py ^
        --image ./low00323.png ^
        --checkpoint ./experiments/Restormer_128_2_60k_MDTA/net_g_44000.pth ^
        --attn_type MDTA --level 1 --block 0

参数:
    --image       必选，输入图像路径
    --checkpoint  必选，模型权重路径
    --attn_type   必选，注意力类型: MDTA | HTA | WTA
    --level       Encoder 层级 1-3 或 Latent=4，默认 1
    --block       Block 索引，-1 表示最后一个，默认 -1
    --size        resize 尺寸 (需配合 --resize)，默认 128
    --resize      是否 resize 输入图像，默认不 resize
    --output_dir  输出目录，默认 visualization/encoder

输出:
    {output_dir}/{图片名}/{level_name}/{attn_type}/block{N}/
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from common import (
    OverlapPatchEmbed,
    Downsample,
    TransformerBlockWithMap,
    load_image,
    compute_l2_norm,
    get_level_name,
    save_figure,
    load_checkpoint,
    load_module_from_state_dict,
    attn_v_to_4d,
)


# ============== 配置 ==============

LEVEL_CONFIG = {
    1: {"dim": 48, "heads": 1, "num_blocks": 4, "prefix": "encoder_level1"},
    2: {"dim": 96, "heads": 2, "num_blocks": 6, "prefix": "encoder_level2"},
    3: {"dim": 192, "heads": 4, "num_blocks": 6, "prefix": "encoder_level3"},
    4: {"dim": 384, "heads": 8, "num_blocks": 8, "prefix": "latent"},
}

ATTENTION_TITLES = {
    "MDTA": "MDTA (Channel: CxC)",
    "HTA": "HTA (Column: WxW)",
    "WTA": "WTA (Row: HxH)",
}

SIMPLE_DIR = (
    os.path.join("visualization", "encoder"),
    os.path.join("visualization", "encoder_no_anno"),
)


# ============== 完整 Encoder 可视化模型 ==============


class RestormerEncoderVisualizer(torch.nn.Module):
    """
    Restormer Encoder 可视化模型
    支持 Level 1-4，串行执行到指定 level 和 block
    """

    def __init__(self, attn_type="MDTA", bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.attn_type = attn_type

        self.patch_embed = OverlapPatchEmbed(in_c=3, embed_dim=48, bias=bias)

        self.encoder_level1 = torch.nn.ModuleList(
            [
                TransformerBlockWithMap(48, 1, 2.66, bias, LayerNorm_type, attn_type)
                for _ in range(4)
            ]
        )
        self.down1_2 = Downsample(48)

        self.encoder_level2 = torch.nn.ModuleList(
            [
                TransformerBlockWithMap(96, 2, 2.66, bias, LayerNorm_type, attn_type)
                for _ in range(6)
            ]
        )
        self.down2_3 = Downsample(96)

        self.encoder_level3 = torch.nn.ModuleList(
            [
                TransformerBlockWithMap(192, 4, 2.66, bias, LayerNorm_type, attn_type)
                for _ in range(6)
            ]
        )
        self.down3_4 = Downsample(192)

        self.latent = torch.nn.ModuleList(
            [
                TransformerBlockWithMap(384, 8, 2.66, bias, LayerNorm_type, attn_type)
                for _ in range(8)
            ]
        )

        self.levels = {
            1: self.encoder_level1,
            2: self.encoder_level2,
            3: self.encoder_level3,
            4: self.latent,
        }
        self.downsamples = {1: self.down1_2, 2: self.down2_3, 3: self.down3_4}

    def forward(self, x, target_level=1, target_block=0):
        """
        串行执行到 target_level 的 target_block
        返回: patch_feat, level_input, block_input, block_output, attn_map, attn_output, value_map, attn_v_map
        """
        x = self.patch_embed(x)
        patch_feat = x.clone()

        level_input = block_input = block_output = None
        attn_map = attn_output = value_map = attn_v_map = None

        for level in range(1, target_level + 1):
            if level == target_level:
                level_input = x.clone()

            blocks = self.levels[level]
            blocks_to_run = (target_block + 1) if level == target_level else len(blocks)

            for i in range(blocks_to_run):
                if level == target_level and i == target_block:
                    block_input = x.clone()

                x = blocks[i](x)

                if level == target_level and i == target_block:
                    block_output = x.clone()
                    attn_map = blocks[i].get_attn_map()
                    attn_output = blocks[i].get_attn_output()
                    value_map = blocks[i].get_value_map()
                    attn_v_map = blocks[i].get_attn_v_map()

            if level < target_level and level in self.downsamples:
                x = self.downsamples[level](x)

        return (
            patch_feat,
            level_input,
            block_input,
            block_output,
            attn_map,
            attn_output,
            value_map,
            attn_v_map,
        )

    def load_from_checkpoint(self, checkpoint_path, device):
        """从 Restormer checkpoint 加载权重"""
        state_dict = load_checkpoint(checkpoint_path, device)
        if state_dict is None:
            return False

        load_module_from_state_dict(
            state_dict, "patch_embed.", self.patch_embed, "patch_embed"
        )
        load_module_from_state_dict(state_dict, "down1_2.", self.down1_2, "down1_2")
        load_module_from_state_dict(state_dict, "down2_3.", self.down2_3, "down2_3")
        load_module_from_state_dict(state_dict, "down3_4.", self.down3_4, "down3_4")

        level_prefixes = {
            1: "encoder_level1",
            2: "encoder_level2",
            3: "encoder_level3",
            4: "latent",
        }
        for level, prefix in level_prefixes.items():
            blocks = self.levels[level]
            for i, block in enumerate(blocks):
                load_module_from_state_dict(
                    state_dict, f"{prefix}.{i}.", block, f"{prefix}.{i}"
                )
            print(f"Loaded: {prefix} ({len(blocks)} blocks)")

        return True


# ============== 可视化函数 ==============


def save_attention_map(attn_type, attn_map, save_dir, level, block):
    attn_np = (
        attn_map[0, 0].cpu().numpy()
        if attn_map.dim() == 4
        else attn_map[0, 0, 0].cpu().numpy()
    )
    level_name = get_level_name(level)

    save_figure(
        attn_np,
        save_dir,
        "attention_map.png",
        simple_dir_replace=SIMPLE_DIR,
        title=f"{ATTENTION_TITLES[attn_type]}\n{level_name} Block {block} | Shape: {attn_np.shape}",
        cmap="inferno",
        aspect="auto",
        xlabel="Key",
        ylabel="Query",
    )


def save_feature_map(
    attn_type,
    input_img,
    patch_feat,
    level_input,
    block_input,
    block_output,
    save_dir,
    level,
    block,
):
    os.makedirs(save_dir, exist_ok=True)
    level_name = get_level_name(level)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    img_np = np.clip(input_img[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Input Image", fontsize=10)
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(patch_feat[0].cpu().mean(dim=0).numpy(), cmap="viridis")
    axes[0, 1].set_title(f"Patch Embed\n{list(patch_feat.shape)}", fontsize=10)
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow(level_input[0].cpu().mean(dim=0).numpy(), cmap="viridis")
    axes[0, 2].set_title(f"{level_name} Input\n{list(level_input.shape)}", fontsize=10)
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(block_input[0].cpu().mean(dim=0).numpy(), cmap="viridis")
    axes[1, 0].set_title(f"Block {block} Input", fontsize=10)
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(block_output[0].cpu().mean(dim=0).numpy(), cmap="viridis")
    axes[1, 1].set_title(f"Block {block} Output", fontsize=10)
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    diff = (block_output[0] - block_input[0]).abs().cpu().mean(dim=0).numpy()
    im = axes[1, 2].imshow(diff, cmap="magma")
    axes[1, 2].set_title(f"Block {block} |Output - Input|", fontsize=10)
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"{ATTENTION_TITLES[attn_type]} - {level_name} Block {block}", fontsize=12
    )
    plt.tight_layout()

    save_path = os.path.join(save_dir, "feature_map.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def save_value_map(attn_type, value_map, save_dir, level, block):
    """保存 Value (V) 的可视化"""
    v_np = (
        value_map[0, 0].cpu().numpy()
        if value_map.dim() == 4
        else value_map[0, 0, 0].cpu().numpy()
    )
    level_name = get_level_name(level)

    # 根据注意力类型设置标签
    label_config = {
        "MDTA": (
            f"Value Matrix (V)\nShape: {v_np.shape} [channels × spatial]",
            "Spatial Dimension (H×W)",
            "Channel Dimension",
        ),
        "HTA": (
            f"Value Matrix (V)\nShape: {v_np.shape} [width × features]",
            "Feature Dimension (C×H)",
            "Width (W)",
        ),
        "WTA": (
            f"Value Matrix (V)\nShape: {v_np.shape} [height × features]",
            "Feature Dimension (C×W)",
            "Height (H)",
        ),
    }
    title, xlabel, ylabel = label_config[attn_type]

    save_figure(
        v_np,
        save_dir,
        "value_map.png",
        simple_dir_replace=SIMPLE_DIR,
        title=title,
        suptitle=f"{ATTENTION_TITLES[attn_type]} - {level_name} Block {block}",
        cmap="plasma",
        aspect="auto",
        xlabel=xlabel,
        ylabel=ylabel,
    )


def save_attn_v_map(attn_type, attn_v_map, block_input, save_dir, level, block):
    """保存 Attention Output (BEFORE project_out) 的可视化 - 使用 L2 Norm"""
    num_heads = LEVEL_CONFIG[level]["heads"]
    attn_v_4d = attn_v_to_4d(attn_type, attn_v_map, block_input, num_heads)
    l2_norm = compute_l2_norm(attn_v_4d)
    level_name = get_level_name(level)

    save_figure(
        l2_norm,
        save_dir,
        "attn_output_before_proj.png",
        simple_dir_replace=SIMPLE_DIR,
        title="Attention Output (L2 Norm)\nBEFORE project_out (Fixed Reshape)",
        suptitle=f"{ATTENTION_TITLES[attn_type]} - {level_name} Block {block}",
    )


def save_attention_output(attn_output, save_dir, level, block):
    """保存 Attention Output (AFTER project_out，不含残差和FFN) - 使用 L2 Norm"""
    l2_norm = compute_l2_norm(attn_output)
    level_name = get_level_name(level)

    save_figure(
        l2_norm,
        save_dir,
        "attn_output_after_proj.png",
        simple_dir_replace=SIMPLE_DIR,
        title="L2 Norm across channels",
        suptitle=f"{level_name} Block {block} Attention Output\n(AFTER project_out, no residual/FFN)",
    )


def save_block_output_feature(block_output, save_dir, level, block):
    """单独保存 block 输出的特征图 - 使用 L2 Norm"""
    l2_norm = compute_l2_norm(block_output)
    level_name = get_level_name(level)

    save_figure(
        l2_norm,
        save_dir,
        "block_output_feature.png",
        simple_dir_replace=SIMPLE_DIR,
        title="L2 Norm across channels",
        suptitle=f"{level_name} Block {block} Output Feature (FFN Output)\nShape: {list(block_output.shape)}",
    )

    # 同时保存原始 tensor 以便后续分析
    os.makedirs(save_dir, exist_ok=True)
    tensor_path = os.path.join(save_dir, "block_output_feature.pt")
    torch.save(block_output.cpu(), tensor_path)
    print(f"Saved: {tensor_path}")


def save_channel_attention_maps(
    attn_type, attn_map, save_dir, level, block, num_channels=8
):
    if attn_map.dim() != 5:
        return

    os.makedirs(save_dir, exist_ok=True)
    c = min(num_channels, attn_map.shape[2])
    level_name = get_level_name(level)

    for simple in [False, True]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i in range(c):
            attn_np = attn_map[0, 0, i].cpu().numpy()
            axes[i].imshow(attn_np, cmap="inferno", aspect="auto")
            if not simple:
                axes[i].set_title(f"Channel {i}", fontsize=10)
            axes[i].axis("off")

        for i in range(c, len(axes)):
            axes[i].axis("off")

        if not simple:
            plt.suptitle(
                f"{attn_type} {level_name} Block {block} - Per Channel", fontsize=12
            )
        plt.tight_layout()

        if simple:
            out_dir = save_dir.replace(SIMPLE_DIR[0], SIMPLE_DIR[1])
        else:
            out_dir = save_dir
        os.makedirs(out_dir, exist_ok=True)

        save_path = os.path.join(out_dir, "channel_attention_maps.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        label = " (simple)" if simple else ""
        print(f"Saved{label}: {save_path}")


# ============== 主函数 ==============


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Restormer Attention (Multi-Level)"
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--attn_type",
        type=str,
        required=True,
        choices=["MDTA", "HTA", "WTA"],
        help="Attention type",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Encoder level (1-3) or Latent (4)",
    )
    parser.add_argument(
        "--block", type=int, default=-1, help="Block index (-1 for last block)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Image size (only used when --resize is set)",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Resize image to --size (default: no resize)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join("visualization", "encoder"), help="Output directory"
    )
    args = parser.parse_args()

    # 处理 block 索引（-1 表示最后一个）
    max_blocks = LEVEL_CONFIG[args.level]["num_blocks"]
    if args.block == -1:
        args.block = max_blocks - 1
    elif args.block >= max_blocks:
        print(
            f"Error: Level {args.level} only has {max_blocks} blocks (0-{max_blocks - 1})"
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    level_name = "latent" if args.level == 4 else f"encoder{args.level}"

    print(f"Device: {device}")
    print(f"Attention: {args.attn_type}")
    print(f"Level: {args.level} ({level_name}), Block: {args.block}")
    print(f"Checkpoint: {args.checkpoint}")

    # 创建模型
    model = RestormerEncoderVisualizer(attn_type=args.attn_type).to(device)

    if not model.load_from_checkpoint(args.checkpoint, device):
        return
    model.eval()

    # 加载图像
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
    img_tensor = load_image(args.image, args.size, no_resize=not args.resize).to(device)
    print(f"Image: {args.image} -> {img_tensor.shape}")
    if args.resize:
        print(f"Resized to: {args.size}x{args.size}")
    else:
        print("No resize (original size)")

    # 前向传播
    with torch.no_grad():
        (
            patch_feat,
            level_input,
            block_input,
            block_output,
            attn_map,
            attn_output,
            value_map,
            attn_v_map,
        ) = model(img_tensor, target_level=args.level, target_block=args.block)

    print(f"\nPatch feat: {patch_feat.shape}")
    print(f"Level {args.level} input: {level_input.shape}")
    print(f"Block {args.block} input: {block_input.shape}")
    print(f"Block {args.block} output: {block_output.shape}")
    print(f"Attention map: {attn_map.shape}")
    print(f"Attention output: {attn_output.shape}")
    print(f"Value map: {value_map.shape}")
    print(f"Attention*V map: {attn_v_map.shape}")

    # 保存结果 - 目录结构: visualization/图片文件名/encoder1/MDTA/block0
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    save_dir = os.path.join(
        args.output_dir, image_name, level_name, args.attn_type, f"block{args.block}"
    )

    save_attention_map(args.attn_type, attn_map, save_dir, args.level, args.block)
    save_value_map(args.attn_type, value_map, save_dir, args.level, args.block)
    save_attn_v_map(
        args.attn_type, attn_v_map, block_input, save_dir, args.level, args.block
    )
    save_attention_output(attn_output, save_dir, args.level, args.block)
    save_feature_map(
        args.attn_type,
        img_tensor,
        patch_feat,
        level_input,
        block_input,
        block_output,
        save_dir,
        args.level,
        args.block,
    )
    save_block_output_feature(block_output, save_dir, args.level, args.block)
    save_channel_attention_maps(
        args.attn_type, attn_map, save_dir, args.level, args.block
    )

    print(f"\nDone! Results: {save_dir}/")


if __name__ == "__main__":
    main()
