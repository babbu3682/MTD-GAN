import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random

from .utils.grad_loss import SobelOperator
from .utils.gan_loss import ls_gan
from .utils.ops import turn_on_spectral_norm
from .DUGAN_wrapper import UNet
from .REDCNN.REDCNN_wrapper import Generator


# Reference : https://github.com/Hzzone/DU-GAN/blob/master/models/DUGAN/DUGAN.py


class DUGAN(nn.Module):
    def __init__(self):
        super(DUGAN, self).__init__()
        self.Generator            = Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)
        self.Image_Discriminator  = turn_on_spectral_norm(UNet(repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False))
        self.Grad_Discriminator   = copy.deepcopy(self.Image_Discriminator)

        self.sobel              = SobelOperator().cuda()
        self.gan_metric         = ls_gan

    def Image_d_loss(self, x, y):
        fake                   = self.Generator(x).detach()   # fixed this line.
        real_enc,  real_dec    = self.Image_Discriminator(y)
        fake_enc,  fake_dec    = self.Image_Discriminator(fake)
        input_enc, input_dec   = self.Image_Discriminator(x)
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + \
                    self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.) + \
                    self.gan_metric(input_enc, 0.) + self.gan_metric(input_dec, 0.)

        total_loss = disc_loss

        # CutMix...
        msg_dict = {}
        if random.random() >= 0.5:
            mask = cutmix(real_dec.size()).to(real_dec)
            cutmix_enc, cutmix_dec = self.Image_Discriminator(mask_src_tgt(y, fake, mask))
            cutmix_disc_loss       = self.gan_metric(cutmix_enc, 0.) + self.gan_metric(cutmix_dec, mask)
            cr_loss                = F.mse_loss(cutmix_dec, mask_src_tgt(real_dec, fake_dec, mask))

            total_loss             += cutmix_disc_loss + 1.0*cr_loss

            msg_dict.update({
                'D/Img_cutmix_loss': cutmix_disc_loss,
                'D/Img_cr_loss': cr_loss,
            })

        return total_loss, msg_dict

    def Grad_d_loss(self, x, y):
        grad_fake                        = self.sobel(self.Generator(x)).detach()   # fixed this line.
        grad_real_enc,   grad_real_dec   = self.Grad_Discriminator(self.sobel(y))
        grad_fake_enc,   grad_fake_dec   = self.Grad_Discriminator(grad_fake)
        grad_input_enc,  grad_input_dec  = self.Grad_Discriminator(self.sobel(x))

        disc_loss = self.gan_metric(grad_real_enc, 1.) + self.gan_metric(grad_real_dec, 1.) + \
                    self.gan_metric(grad_fake_enc, 0.) + self.gan_metric(grad_fake_dec, 0.) + \
                    self.gan_metric(grad_input_enc, 0.) + self.gan_metric(grad_input_dec, 0.)
        
        total_loss = disc_loss

        # CutMix...!
        msg_dict = {}
        if random.random() >= 0.5:
            mask = cutmix(grad_real_dec.size()).to(grad_real_dec)
            cutmix_enc, cutmix_dec = self.Grad_Discriminator(mask_src_tgt(self.sobel(y), grad_fake, mask))
            cutmix_disc_loss       = self.gan_metric(cutmix_enc, 0.) + self.gan_metric(cutmix_dec, mask)
            cr_loss                = F.mse_loss(cutmix_dec, mask_src_tgt(grad_real_dec, grad_fake_dec, mask))

            total_loss             += cutmix_disc_loss + 1.0*cr_loss

            msg_dict.update({
                'D/Grad_cutmix_loss': cutmix_disc_loss,
                'D/Grad_cr_loss': cr_loss,
            })

        return total_loss, msg_dict

    def g_loss(self, x, y):
        fake                        = self.Generator(x)
        img_gen_enc,  img_gen_dec   = self.Image_Discriminator(fake)
        grad_gen_enc, grad_gen_dec  = self.Grad_Discriminator(self.sobel(fake))
        
        img_gen_loss                = self.gan_metric(img_gen_enc, 1.) + self.gan_metric(img_gen_dec, 1.)
        grad_gen_loss               = self.gan_metric(grad_gen_enc, 1.) + self.gan_metric(grad_gen_dec, 1.)

        adv_loss  = 0.1*img_gen_loss + 0.1*grad_gen_loss 
        pix_loss  = 1.0*F.mse_loss(fake, y)
        grad_loss = 20.0*F.l1_loss(self.sobel(fake), self.sobel(y))

        total_loss = adv_loss + pix_loss + grad_loss
        msg_dict = {}
        msg_dict.update({
            'G/adv_loss': adv_loss,
            'G/pix_loss': pix_loss,
            'G/grad_loss': grad_loss,
        })

        return total_loss, msg_dict
        



def cutmix(mask_size):
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
    mask = torch.ones(mask_size)
    mask[:, :, y0:y1, x0:x1] = 0
    return mask

def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target









