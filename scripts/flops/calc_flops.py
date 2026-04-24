"""
计算 Restormer 模型的 FLOPs 和参数量
========================================
功能:
    使用 torch.utils.flop_counter.FlopCounterMode 计算 Restormer 模型在不同输入
    尺寸下的浮点运算量 (FLOPs) 和参数量，结果追加写入项目根目录的 Flops&Params.txt。

使用方法:
    python scripts/flops/calc_flops.py

说明:
    - 无需命令行参数，输入尺寸在脚本内 input_sizes 变量中配置
    - 默认计算 (3, 400, 600) 和 (3, 256, 256) 两种输入尺寸
    - 自动检测 GPU，优先使用 CUDA 加速
    - 结果同时打印到控制台并追加到 Flops&Params.txt
"""

import sys
import os
import types
import datetime
import importlib.util
import torch
from torch.utils.flop_counter import FlopCounterMode

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARCHS_DIR = os.path.join(PROJECT_ROOT, "basicsr", "models", "archs")


def _load_module_from_file(module_name, file_path):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Register a fake parent package so relative imports in restormer_arch.py resolve correctly
pkg_name = "basicsr.models.archs"
for partial in ["basicsr", "basicsr.models", pkg_name]:
    if partial not in sys.modules:
        pkg = types.ModuleType(partial)
        pkg.__path__ = [ARCHS_DIR] if partial == pkg_name else []
        pkg.__package__ = partial
        sys.modules[partial] = pkg

# Load the dependency first, then the arch module
_load_module_from_file(
    f"{pkg_name}.extra_attention_raw",
    os.path.join(ARCHS_DIR, "extra_attention_raw.py"),
)
restormer_mod = _load_module_from_file(
    f"{pkg_name}.restormer_arch",
    os.path.join(ARCHS_DIR, "restormer_arch.py"),
)
Restormer = restormer_mod.Restormer


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_flops(flops):
    """Format FLOPs into human-readable string."""
    if flops >= 1e12:
        return f"{flops / 1e12:.4f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.4f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.4f} MFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


def format_params(params):
    """Format parameter count into human-readable string."""
    if params >= 1e6:
        return f"{params / 1e6:.4f} M"
    elif params >= 1e3:
        return f"{params / 1e3:.4f} K"
    else:
        return f"{params}"


def measure_flops(model, input_shape, device):
    """Measure FLOPs for a given input shape."""
    dummy_input = torch.randn(1, *input_shape, device=device)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        with torch.no_grad():
            _ = model(dummy_input)
    return flop_counter.get_total_flops()


def main():
    # Configuration
    input_channels = 3
    input_sizes = [(400, 600), (256, 256)]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = Restormer(
        inp_channels=input_channels,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    )
    model = model.to(device).eval()

    # Count parameters (same for all input sizes)
    total_params, trainable_params = count_parameters(model)

    # Generate report
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        f"Run Time: {now}",
        "Model: Restormer",
        f"Device: {device}",
        "",
        f"Total Params: {total_params:,} ({format_params(total_params)})",
    ]
    if trainable_params != total_params:
        report_lines.append(
            f"Trainable Params: {trainable_params:,} ({format_params(trainable_params)})"
        )

    # Measure FLOPs for each input size
    for h, w in input_sizes:
        flops = measure_flops(model, (input_channels, h, w), device)
        report_lines.append("")
        report_lines.append(f"Input Shape: ({input_channels}, {h}, {w})")
        report_lines.append(f"FLOPs: {flops:,} ({format_flops(flops)})")

    report = "\n".join(report_lines)

    # Print to console
    print(report)

    # Append to Flops&Params.txt
    output_path = os.path.join(PROJECT_ROOT, "Flops&Params.txt")
    with open(output_path, "a", encoding="utf-8") as f:
        # Add separator if file is not empty
        if os.path.getsize(output_path) > 0:
            f.write("\n" + "-" * 50 + "\n")
        f.write(report + "\n")

    print(f"\nResults appended to: {output_path}")


if __name__ == "__main__":
    main()
