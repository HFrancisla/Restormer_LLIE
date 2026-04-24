"""
Restormer 第一层 Block 0 完整流程可视化
========================================
功能:
    可视化 Encoder Level 1 Block 0 的完整数据流: Patch Embedding → Q/K/V →
    Attention Map → Attention*V → Residual → FFN → Output。
    支持 MDTA/HTA/WTA，生成 3×4 总览图和各中间结果的单独高清图。

使用方法:
    python scripts/visualization/visualize_first_block.py ^
        --image ./low00323.png ^
        --checkpoint ./experiments/Restormer_128_2_60k_MDTA/net_g_44000.pth ^
        --attn_type MDTA

参数:
    --image       必选，输入图像路径
    --checkpoint  必选，模型权重路径
    --attn_type   必选，注意力类型: MDTA | HTA | WTA
    --resize      resize 到 256×256，默认不 resize
    --output_dir  输出目录，默认 visualization/first_block
    --cmap        颜色映射，默认 jet

输出:
    {output_dir}/{图片名}/{attn_type}/
        00_overview.png, 01_q_spatial.png, ..., 13_second_residual.png
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from common import (
    LayerNorm,
    FeedForward,
    OverlapPatchEmbed,
    AttentionWithOutputs,
    load_image,
    compute_l2_norm,
    save_figure,
    load_checkpoint,
    load_module_from_state_dict,
)

SIMPLE_DIR = (
    os.path.join("visualization", "first_block"),
    os.path.join("visualization", "first_block_no_anno"),
)


# ============== 简化的 Encoder Level 1 Block 0 模型 ==============


class FirstBlockVisualizer(nn.Module):
    """只包含 Patch Embedding 和 Encoder Level 1 的第一个 Block"""

    def __init__(self, attn_type="MDTA", bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.attn_type = attn_type

        self.patch_embed = OverlapPatchEmbed(in_c=3, embed_dim=48, bias=bias)

        # Encoder Level 1 Block 0 (save_intermediates=True 保存 Q/K/V 等中间结果)
        self.norm1 = LayerNorm(48, LayerNorm_type)
        self.attn = AttentionWithOutputs(
            48, 1, bias, attn_type, save_intermediates=True
        )
        self.norm2 = LayerNorm(48, LayerNorm_type)
        self.ffn = FeedForward(48, ffn_expansion_factor=2.66, bias=bias)

    def forward(self, x):
        patch_feat = self.patch_embed(x)
        block_input = patch_feat.clone()

        # LayerNorm + Attention
        attn_out = self.attn(self.norm1(patch_feat))
        first_residual = block_input + attn_out

        # LayerNorm + FFN
        ffn_out = self.ffn(self.norm2(first_residual))
        second_residual = first_residual + ffn_out

        return {
            "patch_feat": patch_feat,
            "block_input": block_input,
            "q_before_rearrange": self.attn.q_before_rearrange,
            "k_before_rearrange": self.attn.k_before_rearrange,
            "v_before_rearrange": self.attn.v_before_rearrange,
            "q_after_rearrange": self.attn.q_after_rearrange,
            "k_after_rearrange": self.attn.k_after_rearrange,
            "v_after_rearrange": self.attn.v_after_rearrange,
            "attn_map": self.attn.attn_map,
            "attn_v_before_rearrange": self.attn.attn_v_before_rearrange,
            "attn_v_after_rearrange": self.attn.attn_v_after_rearrange,
            "attn_output": attn_out,
            "first_residual": first_residual,
            "ffn_output": ffn_out,
            "second_residual": second_residual,
        }

    def load_from_checkpoint(self, checkpoint_path, device):
        """从 Restormer checkpoint 加载权重"""
        state_dict = load_checkpoint(checkpoint_path, device)
        if state_dict is None:
            return False

        load_module_from_state_dict(
            state_dict, "patch_embed.", self.patch_embed, "patch_embed"
        )

        block_prefix = "encoder_level1.0."
        load_module_from_state_dict(
            state_dict, f"{block_prefix}norm1.", self.norm1, "encoder_level1.0.norm1"
        )
        load_module_from_state_dict(
            state_dict,
            f"{block_prefix}attn.",
            self.attn,
            f"encoder_level1.0.attn ({self.attn_type})",
        )
        load_module_from_state_dict(
            state_dict, f"{block_prefix}norm2.", self.norm2, "encoder_level1.0.norm2"
        )
        load_module_from_state_dict(
            state_dict, f"{block_prefix}ffn.", self.ffn, "encoder_level1.0.ffn"
        )

        return True


# ============== 可视化函数 ==============


def visualize_all(input_img, outputs, attn_type, save_dir, colormap="jet"):
    """可视化所有中间结果（3x4 总览图 + 各个单独图）"""
    os.makedirs(save_dir, exist_ok=True)

    # 预计算所有 L2 Norm（避免重复计算）
    img_np = np.clip(input_img[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
    patch_l2 = compute_l2_norm(outputs["patch_feat"])
    block_l2 = compute_l2_norm(outputs["block_input"])
    q_np = outputs["q_after_rearrange"][0, 0].cpu().numpy()
    k_np = outputs["k_after_rearrange"][0, 0].cpu().numpy()
    v_np = outputs["v_after_rearrange"][0, 0].cpu().numpy()
    attn_np = outputs["attn_map"][0, 0].cpu().numpy()
    attn_v_l2 = compute_l2_norm(outputs["attn_v_after_rearrange"])
    attn_out_l2 = compute_l2_norm(outputs["attn_output"])
    first_res_l2 = compute_l2_norm(outputs["first_residual"])
    ffn_l2 = compute_l2_norm(outputs["ffn_output"])
    second_res_l2 = compute_l2_norm(outputs["second_residual"])

    # === 3x4 总览图 ===
    grid_items = [
        # Row 1
        (img_np, f"Input Image\n{list(input_img.shape)}", None),
        (
            patch_l2,
            f"Patch Embedding (L2 Norm)\n{list(outputs['patch_feat'].shape)}",
            None,
        ),
        (
            block_l2,
            f"Block 0 Input (L2 Norm)\n{list(outputs['block_input'].shape)}",
            None,
        ),
        (
            q_np,
            f"Query (after rearrange)\n{list(outputs['q_after_rearrange'].shape)}",
            "auto",
        ),
        # Row 2
        (
            k_np,
            f"Key (after rearrange)\n{list(outputs['k_after_rearrange'].shape)}",
            "auto",
        ),
        (
            v_np,
            f"Value (after rearrange)\n{list(outputs['v_after_rearrange'].shape)}",
            "auto",
        ),
        (attn_np, f"Attention Map\n{list(outputs['attn_map'].shape)}", "auto"),
        (
            attn_v_l2,
            f"Attention*V (L2 Norm)\nAFTER rearrange BEFORE project_out\n{list(outputs['attn_v_after_rearrange'].shape)}",
            None,
        ),
        # Row 3
        (
            attn_out_l2,
            f"Attention Output (L2 Norm)\n{list(outputs['attn_output'].shape)}",
            None,
        ),
        (
            first_res_l2,
            f"First Residual (L2 Norm)\nInput + Attn\n{list(outputs['first_residual'].shape)}",
            None,
        ),
        (ffn_l2, f"FFN Output (L2 Norm)\n{list(outputs['ffn_output'].shape)}", None),
        (
            second_res_l2,
            f"Second Residual (L2 Norm)\nFirst Res + FFN\n{list(outputs['second_residual'].shape)}",
            None,
        ),
    ]

    for simple in [False, True]:
        fig = plt.figure(figsize=(20, 15))
        for idx, (data, title, aspect) in enumerate(grid_items):
            ax = plt.subplot(3, 4, idx + 1)
            if idx == 0:
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap=colormap, aspect=aspect)
                if not simple:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if not simple:
                ax.set_title(title, fontsize=10)
            ax.axis("off")

        if not simple:
            plt.suptitle(
                f"{attn_type} - Encoder Level 1 Block 0 Visualization (with Residuals)",
                fontsize=14,
                y=0.98,
            )
            plt.tight_layout()
        else:
            plt.tight_layout(pad=0)

        if simple:
            out_dir = save_dir.replace(SIMPLE_DIR[0], SIMPLE_DIR[1])
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(
                out_dir, "00_overview.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
        else:
            save_path = os.path.join(
                save_dir, "00_overview.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        label = " (simple)" if simple else ""
        print(f"Saved{label}: {save_path}")

    # === 单独保存各个图 ===
    attn_v_before = outputs["attn_v_before_rearrange"][0, 0].cpu().numpy()

    individual_figures = [
        # (data, filename, title, suptitle, aspect, use_save_figure)
        (
            attn_v_before,
            "08_attn_v_raw.png",
            f"{attn_type} - Attention*V\nBEFORE rearrange BEFORE project_out\n{list(outputs['attn_v_before_rearrange'].shape)}",
            None,
            "auto",
            True,
        ),
        (
            attn_v_l2,
            "09_attn_v_spatial.png",
            f"{attn_type} - Attention*V (L2 Norm)\nAFTER rearrange BEFORE project_out\n{list(outputs['attn_v_after_rearrange'].shape)}",
            None,
            None,
            True,
        ),
        (
            attn_out_l2,
            "10_attn_output.png",
            f"{attn_type} - Attention Output (L2 Norm)\nAFTER project_out\n{list(outputs['attn_output'].shape)}",
            None,
            None,
            True,
        ),
        (
            first_res_l2,
            "11_first_residual.png",
            f"{attn_type} - First Residual (L2 Norm)\nInput + Attention\n{list(outputs['first_residual'].shape)}",
            None,
            None,
            True,
        ),
        (
            ffn_l2,
            "12_ffn_output.png",
            f"{attn_type} - FFN Output (L2 Norm)\n{list(outputs['ffn_output'].shape)}",
            None,
            None,
            True,
        ),
        (
            second_res_l2,
            "13_second_residual.png",
            f"{attn_type} - Second Residual (L2 Norm)\nFirst Residual + FFN\n{list(outputs['second_residual'].shape)}",
            None,
            None,
            True,
        ),
    ]

    for data, filename, title, suptitle, aspect, _ in individual_figures:
        save_figure(
            data,
            save_dir,
            filename,
            simple_dir_replace=SIMPLE_DIR,
            title=title,
            suptitle=suptitle,
            cmap=colormap,
            aspect=aspect,
        )


def save_individual_maps(outputs, attn_type, save_dir, colormap="jet"):
    """单独保存每个 map 的高清版本"""
    # Q, K, V (before rearrange - 保留空间结构, L2 Norm)
    qkv_spatial = {
        "q_before_rearrange": "01_q_spatial.png",
        "k_before_rearrange": "02_k_spatial.png",
        "v_before_rearrange": "03_v_spatial.png",
    }
    for name, filename in qkv_spatial.items():
        tensor = outputs[name][0].cpu()
        data = compute_l2_norm(tensor)

        save_figure(
            data,
            save_dir,
            filename,
            simple_dir_replace=SIMPLE_DIR,
            title=f"{attn_type} - {name.upper()}\n(L2 Norm, Spatial Structure)\nShape: {list(tensor.shape)}",
            cmap=colormap,
            figsize=(10, 8),
            dpi=200,
        )

    # Q, K, V, Attention Map (after rearrange - 用于计算注意力)
    maps_after = [
        ("q_after_rearrange", "04_q_rearranged.png", outputs["q_after_rearrange"][0, 0].cpu().numpy()),
        ("k_after_rearrange", "05_k_rearranged.png", outputs["k_after_rearrange"][0, 0].cpu().numpy()),
        ("v_after_rearrange", "06_v_rearranged.png", outputs["v_after_rearrange"][0, 0].cpu().numpy()),
        ("attn_map", "07_attn_map.png", outputs["attn_map"][0, 0].cpu().numpy()),
    ]

    for name, filename, data in maps_after:
        save_figure(
            data,
            save_dir,
            filename,
            simple_dir_replace=SIMPLE_DIR,
            title=f"{attn_type} - {name.upper()}\nShape: {data.shape}",
            cmap=colormap,
            aspect="auto",
            figsize=(10, 8),
            dpi=200,
        )


# ============== 主函数 ==============


def main():
    parser = argparse.ArgumentParser(
        description="Visualize First Block of Restormer Encoder Level 1"
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
        "--resize",
        action="store_true",
        help="Resize image to 256x256 (default: keep original size)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("visualization", "first_block"),
        help="Output directory",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="jet",
        help="Colormap for visualization (default: jet)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Attention: {args.attn_type}")
    print(f"Checkpoint: {args.checkpoint}")

    # 创建模型
    model = FirstBlockVisualizer(attn_type=args.attn_type).to(device)

    if not model.load_from_checkpoint(args.checkpoint, device):
        return
    model.eval()

    # 加载图像
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return

    img_tensor = load_image(args.image, no_resize=not args.resize).to(device)
    print(f"Image: {args.image} -> {img_tensor.shape}")

    # 前向传播
    with torch.no_grad():
        outputs = model(img_tensor)

    print("\nOutputs:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {list(val.shape)}")

    # 保存结果
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    save_dir = os.path.join(args.output_dir, image_name, args.attn_type)

    visualize_all(img_tensor, outputs, args.attn_type, save_dir, colormap=args.cmap)
    save_individual_maps(outputs, args.attn_type, save_dir, colormap=args.cmap)

    print(f"\nDone! Results saved to: {save_dir}/")


if __name__ == "__main__":
    main()
