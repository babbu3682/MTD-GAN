from torch.nn import functional as F
import torch
import numpy as np
import copy
import torchvision
import argparse
import tqdm
import torch.nn as nn

from .utils.grad_loss import SobelOperator
from .utils.gan_loss import ls_gan
from .utils.ops import turn_on_spectral_norm
from .utils.metrics import compute_ssim, compute_psnr, compute_rmse

from .DUGAN_wrapper import *
from .REDCNN.REDCNN_wrapper import Generator


# My factory 에 맞게 변형한 버전
class DUGAN(nn.Module):
    def __init__(self):
        super(DUGAN, self).__init__()

        generator               = Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)
        img_discriminator       = UNet(repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False)
        img_discriminator       = turn_on_spectral_norm(img_discriminator)
        grad_discriminator      = copy.deepcopy(img_discriminator)

        self.sobel              = SobelOperator().cuda()
        self.Generator          = generator.cuda()
        
        self.Img_Discriminator  = img_discriminator.cuda()
        self.Grad_Discriminator = grad_discriminator.cuda()

        self.apply_cutmix_prob  = torch.rand(20000)
        self.gan_metric         = ls_gan

    def train_Img_Discriminator(self, full_dose, low_dose, gen_full_dose, prefix='Img', n_iter=0):
        ############## Train Discriminator ###################
        real_enc, real_dec     = self.Img_Discriminator(full_dose)
        fake_enc, fake_dec     = self.Img_Discriminator(gen_full_dose.detach())
        source_enc, source_dec = self.Img_Discriminator(low_dose)
        
        msg_dict = {}

        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + \
                    self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.) + \
                    self.gan_metric(source_enc, 0.) + self.gan_metric(source_dec, 0.)
        total_loss = disc_loss

        apply_cutmix = self.apply_cutmix_prob[n_iter - 1] < warmup(1000, 0.5, n_iter)
        
        if apply_cutmix:
            mask = cutmix(real_dec.size()).to(real_dec)

            cutmix_enc, cutmix_dec = self.Img_Discriminator(mask_src_tgt(full_dose, gen_full_dose.detach(), mask))

            cutmix_disc_loss       = self.gan_metric(cutmix_enc, 0.) + self.gan_metric(cutmix_dec, mask)

            cr_loss                = F.mse_loss(cutmix_dec, mask_src_tgt(real_dec, fake_dec, mask))

            total_loss             += cutmix_disc_loss + 1.0*cr_loss

            msg_dict.update({
                'loss/{}_cutmix_disc'.format(prefix): cutmix_disc_loss,
                'loss/{}_cr'.format(prefix): cr_loss,
            })

        return total_loss, msg_dict


    def train_Grad_Discriminator(self, grad_full_dose, grad_low_dose, grad_gen_full_dose, prefix='Grad', n_iter=0):
        ############## Train Discriminator ###################
        grad_real_enc,   grad_real_dec        = self.Grad_Discriminator(grad_full_dose)
        grad_fake_enc,   grad_fake_dec        = self.Grad_Discriminator(grad_gen_full_dose.detach())
        grad_source_enc, grad_source_dec      = self.Grad_Discriminator(grad_low_dose)

        msg_dict = {}

        disc_loss = self.gan_metric(grad_real_enc, 1.) + self.gan_metric(grad_real_dec, 1.) + \
                    self.gan_metric(grad_fake_enc, 0.) + self.gan_metric(grad_fake_dec, 0.) + \
                    self.gan_metric(grad_source_enc, 0.) + self.gan_metric(grad_source_dec, 0.)
        total_loss = disc_loss

        apply_cutmix = self.apply_cutmix_prob[n_iter - 1] < warmup(1000, 0.5, n_iter)
        if apply_cutmix:
            mask = cutmix(grad_real_dec.size()).to(grad_real_dec)

            cutmix_enc, cutmix_dec = self.Grad_Discriminator(mask_src_tgt(grad_full_dose, grad_gen_full_dose.detach(), mask))

            cutmix_disc_loss       = self.gan_metric(cutmix_enc, 0.) + self.gan_metric(cutmix_dec, mask)

            cr_loss                = F.mse_loss(cutmix_dec, mask_src_tgt(grad_real_dec, grad_fake_dec, mask))

            total_loss             += cutmix_disc_loss + 1.0*cr_loss

            msg_dict.update({
                'loss/{}_cutmix_disc'.format(prefix): cutmix_disc_loss,
                'loss/{}_cr'.format(prefix): cr_loss,
            })

        return total_loss, msg_dict


def warmup(warmup_iter, cutmix_prob, n_iter):
    return min(n_iter * cutmix_prob / warmup_iter, cutmix_prob)

def cutmix(mask_size):
    mask = torch.ones(mask_size)
    lam = np.random.beta(1., 1.)
    _, _, height, width = mask_size
    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))
    mask[:, :, y0:y1, x0:x1] = 0
    return mask

def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target









