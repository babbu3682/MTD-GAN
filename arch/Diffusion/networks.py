import torch
import torch.nn as nn
import torch.nn.functional as F

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler, PNDMScheduler, DPMScheduler, DPMScheduler_Multi
from torch.cuda.amp import GradScaler, autocast

# Reference: https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/2d_ddpm/2d_ddpm_tutorial.ipynb

class DDPM(nn.Module):
    def __init__(self):
        super(DDPM, self).__init__()

        self.diffusion_unet = DiffusionModelUNet(spatial_dims=2, 
                                                 in_channels=2,
                                                 out_channels=1,
                                                 num_channels=(128, 256, 256),
                                                 attention_levels=(False, True, True),
                                                 num_res_blocks=1,
                                                 num_head_channels=256)

        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        self.inferer = DiffusionInferer(self.scheduler)

        # Loss
        self.criterion = F.mse_loss


    def forward(self, x):
        with autocast(enabled=True):
            noise = torch.randn_like(x).to('cuda')
            self.scheduler.set_timesteps(num_inference_steps=1000) 
            output = self.inferer.sample(input_noise=noise, diffusion_model=self.diffusion_unet, scheduler=self.scheduler, conditioning=x, mode='concat', verbose=True)

        return output

class DDIM(nn.Module):
    def __init__(self):
        super(DDIM, self).__init__()

        self.diffusion_unet = DiffusionModelUNet(spatial_dims=2, 
                                                 in_channels=2,
                                                 out_channels=1,
                                                 num_channels=(128, 256, 256),
                                                 attention_levels=(False, True, True),
                                                 num_res_blocks=1,
                                                 num_head_channels=256)

        self.scheduler = DDIMScheduler(num_train_timesteps=1000)
        
        self.inferer = DiffusionInferer(self.scheduler)

        # Loss
        self.criterion = F.mse_loss


    def forward(self, x):
        with autocast(enabled=True):
            noise = torch.randn_like(x).to('cuda')
            self.scheduler.set_timesteps(num_inference_steps=50)
            output = self.inferer.sample(input_noise=noise, diffusion_model=self.diffusion_unet, scheduler=self.scheduler, conditioning=x, mode='concat', verbose=True)

        return output

class PNDM(nn.Module):
    def __init__(self):
        super(PNDM, self).__init__()

        self.diffusion_unet = DiffusionModelUNet(spatial_dims=2, 
                                                 in_channels=2,
                                                 out_channels=1,
                                                 num_channels=(128, 256, 256),
                                                 attention_levels=(False, True, True),
                                                 num_res_blocks=1,
                                                 num_head_channels=256)

        self.scheduler = PNDMScheduler(num_train_timesteps=1000, skip_prk_steps=True)
        
        self.inferer = DiffusionInferer(self.scheduler)

        # Loss
        self.criterion = F.mse_loss


    def forward(self, x):
        with autocast(enabled=True):
            noise = torch.randn_like(x).to('cuda')
            self.scheduler.set_timesteps(num_inference_steps=50)
            output = self.inferer.sample(input_noise=noise, diffusion_model=self.diffusion_unet, scheduler=self.scheduler, conditioning=x, mode='concat', verbose=True)

        return output
    
class DPM(nn.Module):
    def __init__(self):
        super(DPM, self).__init__()

        self.diffusion_unet = DiffusionModelUNet(spatial_dims=2, 
                                                 in_channels=2,
                                                 out_channels=1,
                                                 num_channels=(128, 256, 256),
                                                 attention_levels=(False, True, True),
                                                 num_res_blocks=1,
                                                 num_head_channels=256)

        self.scheduler = DPMScheduler_Multi(num_train_timesteps=1000)
        
        self.inferer = DiffusionInferer(self.scheduler)

        # Loss
        self.criterion = F.mse_loss


    def forward(self, x):
        with autocast(enabled=True):
            noise = torch.randn_like(x).to('cuda')
            self.scheduler.set_timesteps(num_inference_steps=50)
            output = self.inferer.sample(input_noise=noise, diffusion_model=self.diffusion_unet, scheduler=self.scheduler, conditioning=x, mode='concat', verbose=True)

        return output