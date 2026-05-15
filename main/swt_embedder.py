"""
SSDM: Stationary Wavelet Transform & Adaptive Masking Watermark Embedder
=========================================================================
This module contains the core novelties of the SSDM pipeline:

[NOVELTY 1] SWT Decomposition: 
Uses un-decimated Haar wavelets for shift-invariant, strictly O(H*W) 
complexity frequency splitting. Explicitly preserves the LL band to 
maintain visual structural integrity.

[NOVELTY 2] Joint Spatial-Frequency Adaptive Masking: 
Fuses latent spatial variance (JND) with SWT high-frequency energy 
to robustly hide watermarks in complex/textured regions.
"""

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from pytorch_wavelets import DWTForward, DWTInverse

class SSDMWatermark:
    """Base class for SSDM Watermarking."""
    def __init__(self, device, shape=(1, 4, 64, 64), dtype=torch.float32, w_channel=3, w_level=1, generator=None):
        self.device = device
        self.shape = shape
        self.dtype = dtype
        self.w_channel = w_channel
        self.w_level = w_level

        # Initialize SWT (using standard DWT classes with symmetric padding to simulate undecimated behavior if custom SWT isn't available)
        self.swt_forward = DWTForward(J=self.w_level, wave='db1', mode='symmetric').to(device)
        self.swt_inverse = DWTInverse(wave='db1', mode='symmetric').to(device)

    def get_watermarking_mask(self):
        mask = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        mask[:, self.w_channel, :, :] = True
        return mask

class SSDMWatermarkSWT(SSDMWatermark):
    """Advanced Watermark Embedder using SWT and Adaptive Masking."""
    def __init__(self, device, shape=(1, 4, 64, 64), dtype=torch.float32, w_channel=3, w_level=1, alpha=1.0, n_bits=32, generator=None):
        super().__init__(device, shape, dtype, w_channel, w_level, generator)
        self.alpha = alpha
        self.n_bits = n_bits
        
        self.inv_sqrt2 = 1 / (2**0.5)
        self.lo = torch.tensor([self.inv_sqrt2, self.inv_sqrt2], device=device).view(1, 1, 1, 2)
        self.hi = torch.tensor([-self.inv_sqrt2, self.inv_sqrt2], device=device).view(1, 1, 1, 2)
        self._generate_orthogonal_keys(generator)

    def _generate_orthogonal_keys(self, generator):
        """Generates orthogonal spatial keys using Gram-Schmidt."""
        h, w = self.shape[-2], self.shape[-1]
        raw_keys = randn_tensor((self.n_bits, h, w), generator=generator, device=self.device, dtype=self.dtype)
        N, D = self.n_bits, h * w
        flat_keys = raw_keys.view(N, D)
        ortho_keys = torch.zeros_like(flat_keys)
        
        for i in range(N):
            v = flat_keys[i]
            for j in range(i):
                u = ortho_keys[j]
                v = v - (torch.dot(v, u) * u)
            norm_val = torch.norm(v)
            if norm_val > 1e-6:
                v = v / norm_val
            ortho_keys[i] = v
            
        self.keys_spatial = (ortho_keys * (D ** 0.5)).view(self.n_bits, h, w)
        self.target_msg = torch.randint(0, 2, (self.n_bits,), generator=generator, device=self.device).float()

    def swt_decompose(self, x):
        """O(H*W) SWT Decomposition using Haar filters."""
        x_pad_w = F.pad(x, (0, 1, 0, 0), mode='circular')
        L = F.conv2d(x_pad_w, self.lo, stride=1)
        H = F.conv2d(x_pad_w, self.hi, stride=1)

        L_pad_h = F.pad(L, (0, 0, 0, 1), mode='circular')
        LL = F.conv2d(L_pad_h, self.lo.transpose(2, 3), stride=1)
        LH = F.conv2d(L_pad_h, self.hi.transpose(2, 3), stride=1)

        H_pad_h = F.pad(H, (0, 0, 0, 1), mode='circular')
        HL = F.conv2d(H_pad_h, self.lo.transpose(2, 3), stride=1)
        HH = F.conv2d(H_pad_h, self.hi.transpose(2, 3), stride=1)
        return LL, LH, HL, HH

    def swt_reconstruct(self, LL, LH, HL, HH):
        """Inverse SWT Reconstruction."""
        lo_rec, hi_rec = self.lo, torch.tensor([self.inv_sqrt2, -self.inv_sqrt2], device=self.device).view(1, 1, 1, 2)
        
        LL_pad = F.pad(LL, (0, 0, 1, 0), mode='circular')
        LH_pad = F.pad(LH, (0, 0, 1, 0), mode='circular')
        L_rec = F.conv2d(LL_pad, lo_rec.transpose(2, 3), stride=1) + F.conv2d(LH_pad, hi_rec.transpose(2, 3), stride=1)

        HL_pad = F.pad(HL, (0, 0, 1, 0), mode='circular')
        HH_pad = F.pad(HH, (0, 0, 1, 0), mode='circular')
        H_rec = F.conv2d(HL_pad, lo_rec.transpose(2, 3), stride=1) + F.conv2d(HH_pad, hi_rec.transpose(2, 3), stride=1)

        L_pad, H_pad = F.pad(L_rec, (1, 0, 0, 0), mode='circular'), F.pad(H_rec, (1, 0, 0, 0), mode='circular')
        return (F.conv2d(L_pad, lo_rec, stride=1) + F.conv2d(H_pad, hi_rec, stride=1)) * 0.5

    def compute_adaptive_mask(self, latents, attention_map=None):
        """Computes adaptive mask using spatial variance and SWT energy."""
        target_channel = latents[:, [self.w_channel], :, :]
        
        # 1. Spatial Variance (JND)
        mu = F.avg_pool2d(target_channel, kernel_size=3, stride=1, padding=1)
        sq_mu = F.avg_pool2d(target_channel ** 2, kernel_size=3, stride=1, padding=1)
        sigma = torch.sqrt(torch.abs(sq_mu - mu ** 2) + 1e-6)

        def normalize(t):
            B = t.shape[0]
            t_flat = t.view(B, -1)
            t_min = t_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            t_max = t_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            return (t - t_min) / (t_max - t_min + 1e-6)

        mask_spatial = normalize(sigma)

        # 2. SWT Energy
        try:
            _, LH, HL, HH = self.swt_decompose(target_channel)
            mask_energy = normalize(torch.abs(LH) + torch.abs(HL) + torch.abs(HH))
        except Exception:
            mask_energy = mask_spatial

        combined_mask = torch.sigmoid(((0.5 * mask_spatial + 0.5 * mask_energy) - 0.5) * 5)
        return combined_mask

    def inject_watermark(self, latents, attention_map=None, **kwargs):
        """Injects watermark into the SWT high-frequency bands."""
        mask_adaptive = self.compute_adaptive_mask(latents, attention_map)

        msg_signed = 2 * self.target_msg - 1
        pattern_spatial = torch.einsum('n,nhw->hw', msg_signed, self.keys_spatial) / (self.n_bits ** 0.5)
        
        full_pattern = torch.zeros_like(latents)
        full_pattern[:, self.w_channel, :, :] = pattern_spatial

        x, w = latents[:, [self.w_channel], :, :], full_pattern[:, [self.w_channel], :, :]
        LL_x, LH_x, HL_x, HH_x = self.swt_decompose(x)
        _, LH_w, HL_w, HH_w = self.swt_decompose(w)

        alpha_local = self.alpha * (0.5 + 2.0 * mask_adaptive)

        LH_out, HL_out, HH_out = LH_x + alpha_local * LH_w, HL_x + alpha_local * HL_w, HH_x + alpha_local * HH_w
        x_wm = self.swt_reconstruct(LL_x, LH_out, HL_out, HH_out)

        latents_out = latents.clone()
        latents_out[:, self.w_channel, :, :] = x_wm[:, 0, :, :]
        return latents_out

    def one_minus_p_value(self, latents, attention_map=None, **kwargs):
        """Calculates bit accuracy for the detected watermark."""
        with torch.no_grad():
            mask_adaptive = self.compute_adaptive_mask(latents, attention_map)
            target_channel = latents[:, self.w_channel]
            B = target_channel.shape[0]

            keys_spatial_resized = F.interpolate(self.keys_spatial.unsqueeze(0), size=latents.shape[-2:], mode='bilinear', align_corners=False).squeeze(0) if self.keys_spatial.shape[-2:] != latents.shape[-2:] else self.keys_spatial
            
            weighted_keys = keys_spatial_resized.unsqueeze(0).expand(B, -1, -1, -1) * (0.5 + 2.0 * mask_adaptive.expand(-1, self.n_bits, -1, -1))
            scores = (target_channel.reshape(B, -1).unsqueeze(1) * weighted_keys.reshape(B, self.n_bits, -1)).sum(dim=2)
            
            matches = ((scores > 0).float() == self.target_msg).float().sum(dim=1)
            return (matches / self.n_bits).mean().item()



    def compute_loss(self, latents, attention_map=None, **kwargs):
        mask = self.compute_adaptive_mask(latents, attention_map)
        target_channel = latents[:, self.w_channel]
        B = target_channel.shape[0]

        keys_spatial_resized = F.interpolate(self.keys_spatial.unsqueeze(0), size=latents.shape[-2:], mode='bilinear', align_corners=False).squeeze(0) if self.keys_spatial.shape[-2:] != latents.shape[-2:] else self.keys_spatial

        keys_expanded = keys_spatial_resized.unsqueeze(0).expand(B, -1, -1, -1)
        mask_expanded = mask.expand(-1, self.n_bits, -1, -1)
        weighted_keys = keys_expanded * (0.5 + 2.0 * mask_expanded)

        flat_target = target_channel.reshape(B, -1)
        flat_weighted_keys = weighted_keys.reshape(B, self.n_bits, -1)

        scores = (flat_target.unsqueeze(1) * flat_weighted_keys).sum(dim=2)
        target_signs = 2 * self.target_msg - 1
        alignment = scores * target_signs

        return torch.mean(torch.nn.functional.softplus(-alignment))
