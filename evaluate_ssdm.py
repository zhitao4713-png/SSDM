"""
SSDM Evaluation Script
======================
Evaluates the visual quality (PSNR, SSIM, LPIPS) of the SSDM watermarking pipeline.
"""
import argparse
import os
import torch
import math
from PIL import Image
import torchvision.transforms.functional as TF
import lpips

# Import the SSDM refactored SSIM metric
from loss.ssim_metric import compute_ssim

def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def main():
    parser = argparse.ArgumentParser(description="SSDM Image Quality Evaluation")
    parser.add_argument("--orig_path", type=str, required=True, help="Path to the original (ground truth) image")
    parser.add_argument("--wm_path", type=str, required=True, help="Path to the watermarked image")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"
{'='*60}
[SSDM Evaluation] 📊 Visual Quality Assessment
{'='*60}")
    print(f" ├─ Target Original   : {args.orig_path}")
    print(f" ├─ Target Watermarked: {args.wm_path}")

    if not os.path.exists(args.orig_path) or not os.path.exists(args.wm_path):
        print("
[!] Error: One or both image paths do not exist. Please check the paths.")
        return

    # Load and resize images to standard 512x512
    img_orig = Image.open(args.orig_path).convert('RGB').resize((512, 512))
    img_wm = Image.open(args.wm_path).convert('RGB').resize((512, 512))

    # Convert to Tensors [0, 1]
    t_orig = TF.to_tensor(img_orig).unsqueeze(0).to(device)
    t_wm = TF.to_tensor(img_wm).unsqueeze(0).to(device)

    # 1. PSNR Calculation
    psnr_val = calc_psnr(t_orig, t_wm)

    # 2. SSIM Calculation (Using our refactored module)
    with torch.no_grad():
        ssim_val = compute_ssim(t_orig, t_wm).item()

    # 3. LPIPS Calculation (Perceptual loss expects [-1, 1] range)
    loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).to(device)
    t_orig_lpips = t_orig * 2.0 - 1.0
    t_wm_lpips = t_wm * 2.0 - 1.0
    with torch.no_grad():
        lpips_val = loss_fn_vgg(t_orig_lpips, t_wm_lpips).item()

    print(f"
[SSDM Metrics Results]")
    print(f" ├─ PSNR  (↑) : {psnr_val:.4f} dB")
    print(f" ├─ SSIM  (↑) : {ssim_val:.4f}")
    print(f" └─ LPIPS (↓) : {lpips_val:.4f}")
    print(f"{'='*60}
")

if __name__ == "__main__":
    main()
