"""
Decompose input images into DWT subbands (LL, LH, HL, HH) using DWT_2D from torch_wavelets.
Usage:
    python scripts/dwt_decompose.py <input_path> [--output_dir OUTPUT_DIR] [--wavelet WAVELET]

    <input_path> can be a single image file or a directory of images.
    Results are saved into a folder named after each input file (without extension).
"""

import sys
import argparse
from pathlib import Path

import pywt
import torch
import numpy as np
from PIL import Image

# Add project root to sys.path so we can import from basicsr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basicsr.models.archs.torch_wavelets import DWT_2D

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image_as_tensor(img_path: str) -> torch.Tensor:
    """Load an image and convert to (1, C, H, W) float tensor in [0, 1]."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a (1, C, H, W) or (C, H, W) tensor to a PIL Image, clamped to [0, 1]."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def normalize_subband_for_vis(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a high-frequency subband to [0, 1] for visualization (abs + rescale)."""
    t = tensor.abs()
    t_max = t.max()
    if t_max > 0:
        t = t / t_max
    return t


def _get_ll_gain(wavelet: str) -> float:
    """Compute the DC gain of the 2D LL filter (sum of all kernel weights)."""
    w = pywt.Wavelet(wavelet)
    dec_lo = np.array(w.dec_lo[::-1])
    w_ll = np.outer(dec_lo, dec_lo)
    return float(w_ll.sum())


def decompose_image(img_path: str, output_dir: str, wavelet: str = "haar"):
    img_path = Path(img_path)
    stem = img_path.stem
    ext = img_path.suffix

    # Create output folder named after input file
    out_folder = Path(output_dir) / stem
    out_folder.mkdir(parents=True, exist_ok=True)

    # Copy original image into the folder
    original = Image.open(img_path).convert("RGB")
    original.save(out_folder / f"{stem}_original{ext}")

    # Load and decompose
    tensor = load_image_as_tensor(str(img_path))

    # Ensure spatial dims are even
    _, _, H, W = tensor.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    dwt = DWT_2D(wavelet)
    with torch.no_grad():
        out = dwt(tensor)  # (1, 4*C, H/2, W/2)

    # Split into LL, LH, HL, HH — each (1, C, H/2, W/2)
    C = 3
    x_ll = out[:, 0 * C : 1 * C, :, :]
    x_lh = out[:, 1 * C : 2 * C, :, :]
    x_hl = out[:, 2 * C : 3 * C, :, :]
    x_hh = out[:, 3 * C : 4 * C, :, :]

    # Save subbands
    subbands = {
        "LL": x_ll,
        "LH": x_lh,
        "HL": x_hl,
        "HH": x_hh,
    }

    ll_gain = _get_ll_gain(wavelet)

    for name, sb in subbands.items():
        # Raw: direct clamp(0,1), exactly what the network receives (no rescaling)
        img_raw = tensor_to_image(sb)
        img_raw.save(out_folder / f"{stem}_{name}_raw.png")

        # Visualized: rescaled for human viewing
        if name == "LL":
            img_out = tensor_to_image(sb / ll_gain)
        else:
            img_out = tensor_to_image(normalize_subband_for_vis(sb))
        img_out.save(out_folder / f"{stem}_{name}.png")

    print(f"[Done] {img_path.name} -> {out_folder}")


def main():
    parser = argparse.ArgumentParser(description="DWT subband decomposition for images")
    parser.add_argument("input", type=str, help="Input image file or directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root directory (default: same directory as input)",
    )
    parser.add_argument(
        "--wavelet", type=str, default="haar", help="Wavelet type (default: haar)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        output_dir = args.output_dir or str(input_path.parent)
        decompose_image(str(input_path), output_dir, args.wavelet)
    elif input_path.is_dir():
        output_dir = args.output_dir or str(input_path)
        files = sorted(
            [
                f
                for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
            ]
        )
        if not files:
            print(f"No image files found in {input_path}")
            return
        for f in files:
            decompose_image(str(f), output_dir, args.wavelet)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
