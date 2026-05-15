"""
SSDM Structural Similarity (SSIM) Metric
========================================
A specialized, highly efficient PyTorch implementation of the SSIM index
used primarily as a visual quality constraint for the SSDM watermarking pipeline.
"""
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def _compute_gaussian_1d(window_size: int, sigma: float) -> torch.Tensor:
    """Computes a 1D Gaussian kernel."""
    gauss_vals = [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    tensor_gauss = torch.Tensor(gauss_vals)
    return tensor_gauss / tensor_gauss.sum()

def _build_ssim_window(window_size: int, channels: int) -> Variable:
    """Builds a 2D Gaussian window for SSIM computation."""
    kernel_1d = _compute_gaussian_1d(window_size, 1.5).unsqueeze(1)
    kernel_2d = kernel_1d.mm(kernel_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(kernel_2d.expand(channels, 1, window_size, window_size).contiguous())
    return window

def _calculate_ssim_core(img_a: torch.Tensor, img_b: torch.Tensor, window: torch.Tensor, window_size: int, channels: int, average_size: bool = True) -> torch.Tensor:
    """Core computation logic for SSIM."""
    pad_size = window_size // 2

    mu_a = F.conv2d(img_a, window, padding=pad_size, groups=channels)
    mu_b = F.conv2d(img_b, window, padding=pad_size, groups=channels)

    mu_a_sq = mu_a.pow(2)
    mu_b_sq = mu_b.pow(2)
    mu_a_b = mu_a * mu_b

    sigma_a_sq = F.conv2d(img_a * img_a, window, padding=pad_size, groups=channels) - mu_a_sq
    sigma_b_sq = F.conv2d(img_b * img_b, window, padding=pad_size, groups=channels) - mu_b_sq
    sigma_ab = F.conv2d(img_a * img_b, window, padding=pad_size, groups=channels) - mu_a_b

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    num = (2 * mu_a_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = num / den

    if average_size:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)

class SSDM_SSIM(torch.nn.Module):
    """
    PyTorch Module to calculate SSIM.
    Maintains a cached window to accelerate batched processing during SSDM optimization.
    """
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSDM_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = 1
        self.window = _build_ssim_window(window_size, self.channels)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        (_, current_channels, _, _) = img_a.size()

        if current_channels == self.channels and self.window.data.type() == img_a.data.type():
            current_window = self.window
        else:
            current_window = _build_ssim_window(self.window_size, current_channels)
            if img_a.is_cuda:
                current_window = current_window.cuda(img_a.get_device())
            current_window = current_window.type_as(img_a)
            self.window = current_window
            self.channels = current_channels

        return _calculate_ssim_core(img_a, img_b, current_window, self.window_size, current_channels, self.size_average)

def compute_ssim(img_a: torch.Tensor, img_b: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    """Functional interface for SSIM calculation."""
    (_, channels, _, _) = img_a.size()
    window = _build_ssim_window(window_size, channels)
    if img_a.is_cuda:
        window = window.cuda(img_a.get_device())
    window = window.type_as(img_a)
    return _calculate_ssim_core(img_a, img_b, window, window_size, channels, size_average)
