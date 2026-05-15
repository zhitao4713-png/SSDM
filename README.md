# SSDM: Stationary Wavelet Diffusion Watermarking

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) 
*(Note: Replace XXXXXXX with your actual Zenodo DOI after release)*

This repository contains the official implementation of the algorithm proposed in our paper, submitted to **The Visual Computer**. SSDM is an advanced, robust, and imperceptible watermarking pipeline designed specifically for Latent Diffusion Models (e.g., Stable Diffusion).

## Dependencies and Requirements

The pipeline is implemented in PyTorch. To reproduce our environment and results, please install the following dependencies:

```bash
# Python 3.8+
# PyTorch 1.13.1 or higher (with CUDA support)
pip install torch torchvision
pip install diffusers transformers accelerate
pip install git+https://github.com/fbcotter/pytorch_wavelets
pip install lpips scipy pyyaml tqdm matplotlib
```
Alternatively, you can use the provided `config/config.yaml` and standard requirements list to set up the Conda environment.

## Datasets

The associated testing datasets are located in the `input/` directory. These benchmark images are partially selected from the **[MS-COCO](https://cocodataset.org/)** and **[DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb)** datasets for evaluating visual quality (PSNR, SSIM, LPIPS) and watermark robustness.
- **`input/ssdm_test*.jpg`**: Benchmark images from MS-COCO and DiffusionDB.
- Generated watermarked images and attacked samples will be saved automatically in the `output/` directory.

## Key Algorithms Implementation

Our pipeline operates entirely within the latent space and consists of two core novelties, implemented in `main/swt_embedder.py`:

1. **Shift-Invariant SWT Decomposition (`swt_decompose`)**: 
   Unlike standard DWT, SSDM employs un-decimated Haar wavelets. The low-frequency approximation band (LL) is explicitly excluded from embedding to preserve the primary structural and color features of the image, while ensuring $O(H 	imes W)$ linear time complexity.
2. **Joint Spatial-Frequency Adaptive Masking (`compute_adaptive_mask`)**: 
   The embedding strength is locally modulated by a continuous adaptive mask combining spatial variance (JND) and SWT high-frequency energy. This hides watermarks heavily in complex, textured regions while suppressing them in smooth areas.

## Quick Start

```bash
# Run the main generation, embedding, and attack pipeline
python run_ssdm.py

# Evaluate robustness and imperceptibility metrics
python evaluate_ssdm.py
```

## Citation

If you find our code, datasets, or methodology useful in your research, please cite our paper published in **The Visual Computer**:

```bibtex
@article{ssdm_visual_computer_202X,
  title={Your Full Article Title Here},
  author={Your Name and Co-authors},
  journal={The Visual Computer},
  year={202X},
  publisher={Springer}
}
```
