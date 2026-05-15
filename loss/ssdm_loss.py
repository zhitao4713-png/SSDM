"""
SSDM Loss Functions
===================
Custom loss definitions combining Perceptual, SSIM, and Watermark-specific objectives.
"""

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import torch
import torch.nn as nn
from .perceptual_vgg import SSDMPerceptualVGG
from .ssim_metric import SSDM_SSIM

class LossProvider(nn.Module):
    def __init__(self, loss_weights: list, device):
        super(LossProvider, self).__init__()
        self.loss_weights = loss_weights

        self.loss_img, self.loss_w = nn.MSELoss(), nn.L1Loss()
        self.loss_ssim = SSDM_SSIM()

        # add perceptive loss
        loss_percep = SSDMPerceptualVGG(reduction='sum')
        loss_percep.load_state_dict(torch.load('./loss/perceptual_vgg_weights.pth', map_location='cpu'))
        loss_percep = loss_percep.to(device)
        self.loss_per = lambda pred_img, gt_img: loss_percep((1+pred_img)/2.0, (1+gt_img)/2.0)/ pred_img.shape[0]

    def __call__(self, pred_img_tensor, gt_img_tensor, init_latents, wm_pipe, attacked_latents=None, **kwargs):
        # Determine which latents to use for watermark loss (Robustness vs Original)
        latents_for_w_loss = attacked_latents if attacked_latents is not None else init_latents

        # Image Quality Losses (Always on the generated image vs ground truth)
        lossI = self.loss_img(pred_img_tensor, gt_img_tensor)*self.loss_weights[0]
        lossP = self.loss_per(pred_img_tensor, gt_img_tensor)*self.loss_weights[1]
        lossS = (1-self.loss_ssim(pred_img_tensor, gt_img_tensor))*self.loss_weights[2]

        # Watermark Loss (On potentially attacked latents)
        if hasattr(wm_pipe, 'compute_loss'):
            # New Differentiable Loss (Correlation-based)
            # Pass kwargs like attention_map here
            lossW = wm_pipe.compute_loss(latents_for_w_loss, **kwargs) * self.loss_weights[3]
        else:
            # Legacy Logic (FFT/Replacement-based)
            latents_fft = torch.fft.fftshift(torch.fft.fft2(latents_for_w_loss), dim=(-1, -2))
            lossW = self.loss_w(latents_fft[wm_pipe.watermarking_mask], wm_pipe.gt_patch[wm_pipe.watermarking_mask])*self.loss_weights[3]

        loss = lossW + lossI + lossP + lossS
        logging.info(f'[SSDM Opt] Total: {loss.item():.4f} | WM: {lossW.item():.4f} | MSE: {lossI.item():.4f} | VGG: {lossP.item():.4f} | SSIM: {lossS.item():.4f}')
        return loss
