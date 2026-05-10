

import os
import torch
from diffusers import AutoencoderKL
import yaml
import numpy as np
import cv2
import torch.nn as nn


class molgene_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        config_path = "" # stable-diffusion-v1-5/vae/config.json
        self.vae = AutoencoderKL.from_config(config_path)


    def forward(self, x):
        pixels_tensor = x.to(self.vae.device)
        posterior = self.vae.encode(pixels_tensor)
        return posterior.latent_dist.mean
    

if __name__ == "__main__":
    pass

    