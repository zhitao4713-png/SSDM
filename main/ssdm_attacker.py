"""
SSDM Robustness Attack Suite
============================
Implements various image perturbations (crop, noise, blur, jpeg) to test watermark resilience.
"""
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
import os
from skimage.util import random_noise
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm.auto import tqdm
from bm3d import bm3d_rgb


class SSDMAttackerBase:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError


class GaussianBlurAttacker(SSDMAttackerBase):
    def __init__(self, kernel_size=5, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = cv2.imread(img_path)
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
            cv2.imwrite(out_path, img)


class GaussianNoiseAttacker(SSDMAttackerBase):
    def __init__(self, std=0.05):
        self.std = std

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            image = cv2.imread(img_path)
            image = image / 255.0
            # Add Gaussian noise to the image
            noise_sigma = self.std  # Vary this to change the amount of noise
            noisy_image = random_noise(image, mode='gaussian', var=noise_sigma ** 2)
            # Clip the values to [0, 1] range after adding the noise
            noisy_image = np.clip(noisy_image, 0, 1)
            noisy_image = np.array(255 * noisy_image, dtype='uint8')
            cv2.imwrite(out_path, noisy_image)


class BM3DAttacker(SSDMAttackerBase):
    def __init__(self):
        pass

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path).convert('RGB')
            y_est = bm3d_rgb(np.array(img) / 255, 0.1)  # use standard deviation as 0.1, 0.05 also works
            plt.imsave(out_path, np.clip(y_est, 0, 1), cmap='gray', vmin=0, vmax=1)


class JPEGAttacker(SSDMAttackerBase):
    def __init__(self, quality=80):
        self.quality = quality

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path).convert('RGB')
            img.save(out_path, "JPEG", quality=self.quality)


class BrightnessAttacker(SSDMAttackerBase):
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path).convert('RGB')
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.brightness)
            img.save(out_path)


class ContrastAttacker(SSDMAttackerBase):
    def __init__(self, contrast=0.2):
        self.contrast = contrast

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path).convert('RGB')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast)
            img.save(out_path)



class ScaleAttacker(SSDMAttackerBase):
    def __init__(self, scale=0.5):
        self.scale = scale

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            img = img.resize((int(w * self.scale), int(h * self.scale)))
            img.save(out_path)


class CropAttacker(SSDMAttackerBase):
    def __init__(self, crop_size=0.5):
        self.crop_size = crop_size

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            img = img.crop((int(w * self.crop_size), int(h * self.crop_size), w, h))
            img.save(out_path)


class DiffWMAttacker(SSDMAttackerBase):
    def __init__(self, pipe, batch_size=20, noise_step=60, captions={}):
        self.pipe = pipe
        self.BATCH_SIZE = batch_size
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

    def attack(self, image_paths, out_paths, return_latents=False, return_dist=False, multi=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(prompts_buf,
                                   head_start_latents=latents,
                                   head_start_step=50 - max(self.noise_step // 20, 1),
                                   guidance_scale=7.5,
                                   generator=generator, )
                images = images[0]
                for img, out in zip(images, outs_buf):
                    img.save(out)

            if len(self.captions) != 0:
                prompts = []
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    if img_name[:-4] in self.captions:
                        prompts.append(self.captions[img_name[:-4]])
                    else:
                        prompts.append("")
            else:
                prompts = [""] * len(image_paths)

            for (img_path, out_path), prompt in tqdm(zip(zip(image_paths, out_paths), prompts)):
                if os.path.exists(out_path) and not multi:
                    continue
                
                img = Image.open(img_path).convert('RGB')
                img = np.asarray(img) / 255
                img = (img - 0.5) * 2
                img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
                latents = self.pipe.vae.encode(img).latent_dist
                latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
                noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
                if return_dist:
                    return self.pipe.scheduler.add_noise(latents, noise, timestep, return_dist=True)
                latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)
                latents_buf.append(latents)
                outs_buf.append(out_path)
                prompts_buf.append(prompt)
                if len(latents_buf) == self.BATCH_SIZE:
                    batched_attack(latents_buf, prompts_buf, outs_buf)
                    latents_buf = []
                    prompts_buf = []
                    outs_buf = []
                if return_latents:
                    ret_latents.append(latents.cpu())

            if len(latents_buf) != 0:
                batched_attack(latents_buf, prompts_buf, outs_buf)
            if return_latents:
                return ret_latents

class InstructEditAttacker(SSDMAttackerBase):

    def __init__(self, model_id="timbrooks/instruct-pix2pix", prompt="make it anime", device='cuda'):
        from diffusers import StableDiffusionInstructPix2PixPipeline
        import torch
        
        # Load in fp16 to save memory
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        )
        
        # Use CPU offload to save VRAM (requires accelerate)
        if device == 'cuda':
            try:
                self.pipe.enable_model_cpu_offload()
                print("Enabled model CPU offload for InstructPix2Pix.")
            except Exception as e:
                print(f"Could not enable CPU offload: {e}. Moving to {device} directly.")
                self.pipe.to(device)
        else:
            self.pipe.to(device)
            
        self.prompt = prompt
        self.device = device
        print(f"Initialized InstructEditAttacker with prompt: {self.prompt}")

    def attack(self, image_paths, out_paths, multi=False):
        generator = torch.Generator(self.device).manual_seed(42)
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths)):
            if os.path.exists(out_path) and not multi:
                continue
            
            # Load image
            original_image = Image.open(img_path).convert("RGB")
            # Resize to 512x512 for stability with IP2P default
            original_image = original_image.resize((512, 512))
            
            # Perform instruction-based edit
            with torch.no_grad():
                images = self.pipe(
                    self.prompt, 
                    image=original_image, 
                    num_inference_steps=20, 
                    image_guidance_scale=1.5,
                    generator=generator
                ).images
            
            images[0].save(out_path)

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

class InstructPix2PixAttacker(SSDMAttackerBase):
    def __init__(self, prompt="make it a watercolor painting", device="cuda"):
        self.prompt = prompt
        self.device = device
        # Load the InstructPix2Pix model
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, safety_checker=None
        )
        self.pipe.to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in tqdm(zip(image_paths, out_paths), desc="InstructPix2Pix Attack"):
            if os.path.exists(out_path) and not multi:
                continue
            
            init_image = Image.open(img_path).convert("RGB")
            # Execute the semantic edit attack
            out_img = self.pipe(
                self.prompt, 
                image=init_image, 
                num_inference_steps=20, 
                image_guidance_scale=1.5
            ).images[0]
            out_img.save(out_path)
