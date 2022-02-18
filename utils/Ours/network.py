import torch 
import torch.nn as nn
import torch.nn.functional as F


from .DiffAugment_pytorch import DiffAugment
from .Unet_Factory import *
from .DUGAN_wrapper import *
from .edcnn_model import *


class HF_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        super(HF_Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.kernel_weight = nn.Parameter(torch.zeros(size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        for idx in range(out_channels):
            
            # Sobel Filter
            if idx == 0:
                self.kernel_weight[idx, :, 0, :] = -1          
                self.kernel_weight[idx, :, 0, 1] = -2
                self.kernel_weight[idx, :, -1, :] = 1
                self.kernel_weight[idx, :, -1, 1] = 2
            elif idx == 1:
                self.kernel_weight[idx, :, :, 0] = -1
                self.kernel_weight[idx, :, 1, 0] = -2
                self.kernel_weight[idx, :, :, -1] = 1
                self.kernel_weight[idx, :, 1, -1] = 2
            elif idx == 2:
                self.kernel_weight[idx, :, 0, 0] = -2
                self.kernel_weight[idx, :, 0, 1] = -1
                self.kernel_weight[idx, :, 1, 0] = -1
                self.kernel_weight[idx, :, 1, -1] = 1
                self.kernel_weight[idx, :, -1, 1] = 1
                self.kernel_weight[idx, :, -1, -1] = 2
            elif idx == 3:
                self.kernel_weight[idx, :, 0, 1] = 1
                self.kernel_weight[idx, :, 0, -1] = 2
                self.kernel_weight[idx, :, 1, 0] = -1
                self.kernel_weight[idx, :, 1, -1] = 1
                self.kernel_weight[idx, :, -1, 0] = -2
                self.kernel_weight[idx, :, -1, 1] = -1

            # High Frequency (Image - Blur)
            elif idx == 4:
                self.kernel_weight[idx, :, :, :] = -1/16
                self.kernel_weight[idx, :, 1, :] = -2/16
                self.kernel_weight[idx, :, :, 1] = -2/16
                self.kernel_weight[idx, :, 1, 1] = 12/16      

            # Laplacian or Unsharped mask filter or point edge filter
            elif idx == 5:
                self.kernel_weight[idx, :, 1, :] = -1           
                self.kernel_weight[idx, :, :, 1] = -1
                self.kernel_weight[idx, :, 1, 1] = 4
            elif idx == 6:
                self.kernel_weight[idx, :, :, :] = -1           
                self.kernel_weight[idx, :, 1, 1] += 9   
            elif idx == 7:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 1, :] = -2
                self.kernel_weight[idx, :, :, 1] = -2
                self.kernel_weight[idx, :, 1, 1] = 4

            # Compass Prewitt
            elif idx == 8:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 0, :] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 9:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 0:2, 1:3] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 10:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, :, 2] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 11:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 1:3, 1:3] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 12:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 1, 1] = -2
                self.kernel_weight[idx, :, 2, :] = -1    
            elif idx == 13:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 1:3, 0:2] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 14:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, :, 0] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 15:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, :2, :2] = -1
                self.kernel_weight[idx, :, 1, 1] = -2

            # Line filter
            elif idx == 16:
                self.kernel_weight[idx, :, :, :] = -1           
                self.kernel_weight[idx, :, 1, :] = 2
            elif idx == 17:
                self.kernel_weight[idx, :, :, :] = -1           
                self.kernel_weight[idx, :, :, 1] = 2
            elif idx == 18:
                self.kernel_weight[idx, :, :, :] = -1           
                self.kernel_weight[idx, :, 0, 2] = 2
                self.kernel_weight[idx, :, 1, 1] = 2
                self.kernel_weight[idx, :, 2, 0] = 2
            elif idx == 19:
                self.kernel_weight[idx, :, :, :] = -1           
                self.kernel_weight[idx, :, 0, 0] = 2
                self.kernel_weight[idx, :, 1, 1] = 2
                self.kernel_weight[idx, :, 2, 2] = 2
                                             

        # Define the trainable sobel factor
        if requires_grad:
            self.kernel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32), requires_grad=True)
        else:
            self.kernel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        # if torch.cuda.is_available():
        #     self.kernel_factor = self.kernel_factor.cuda()
        #     if isinstance(self.bias, nn.Parameter):
        #         self.bias = self.bias.cuda()

        kernel_weight = self.kernel_weight * self.kernel_factor

        # if torch.cuda.is_available():
        #     kernel_weight = kernel_weight.cuda()

        out = F.conv2d(x, kernel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return torch.cat([out, x], dim=1)

class LF_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        super(LF_Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.kernel_weight = nn.Parameter(torch.zeros(size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        for idx in range(out_channels):
            
            # Box Blur
            if idx == 0:
                self.kernel_weight[idx, :, :, :] = 1/9

            # Gaussian Blur
            elif idx == 1:
                self.kernel_weight[idx, :, :, :] = 1/16
                self.kernel_weight[idx, :, 1, :] = 2/16
                self.kernel_weight[idx, :, :, 1] = 2/16
                self.kernel_weight[idx, :, 1, 1] = 4/16      

        # Define the trainable sobel factor
        if requires_grad:
            self.kernel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32), requires_grad=True)
        else:
            self.kernel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        kernel_weight = self.kernel_weight * self.kernel_factor
        out = F.conv2d(x, kernel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return torch.cat([out, x], dim=1)

class Low_UNet(UNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lf_conv = LF_Conv(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, input):
        input = self.lf_conv(input)
        
        x = input
        residuals = []

        for i in range(len(self.down_blocks)):
            x, unet_res = self.down_blocks[i](x)
            residuals.append(unet_res)

        bottom_x = self.conv(x) + x
        x = bottom_x
        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)
        dec_out = self.conv_out(x)

        if self.use_discriminator:
            enc_out = self.to_logit(bottom_x)
            if self.use_sigmoid:
                dec_out = torch.sigmoid(dec_out)
                enc_out = torch.sigmoid(enc_out)
            return enc_out.squeeze(), dec_out

        if self.skip_connection:
            dec_out += input
        if self.use_tanh:
            dec_out = torch.tanh(dec_out)

        return dec_out

class High_UNet(UNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hf_conv = HF_Conv(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, input):
        input = self.hf_conv(input)
        
        x = input
        residuals = []

        for i in range(len(self.down_blocks)):
            x, unet_res = self.down_blocks[i](x)
            residuals.append(unet_res)

        bottom_x = self.conv(x) + x
        x = bottom_x
        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)
        dec_out = self.conv_out(x)

        if self.use_discriminator:
            enc_out = self.to_logit(bottom_x)
            if self.use_sigmoid:
                dec_out = torch.sigmoid(dec_out)
                enc_out = torch.sigmoid(enc_out)
            return enc_out.squeeze(), dec_out

        if self.skip_connection:
            dec_out += input
        if self.use_tanh:
            dec_out = torch.tanh(dec_out)

        return dec_out

# KL Divergence loss used in VAE with an image encoder
class KLDLoss(torch.nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class SPADE(nn.Module):
    def __init__(self, norm_type, norm_nc, label_nc):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'syncbatch':
            self.param_free_norm = nn.SyncBatchNorm(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' %norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma  = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta   = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bicubic')
        actv   = self.mlp_shared(segmap)
        gamma  = self.mlp_gamma(actv)
        beta   = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADE_ConvMixer_Block(nn.Module):
    def __init__(self, dim, kernel_size):
        super(SPADE_ConvMixer_Block, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding="same")
        self.gelu1 = nn.GELU()
        self.norm1 = SPADE(norm_type='instance', norm_nc=dim, label_nc=1)

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=1, padding=0)
        self.gelu2 = nn.GELU()
        self.norm2 = SPADE(norm_type='instance', norm_nc=dim, label_nc=1)

        # Initialize by xavier_uniform_
        self.init_weight()
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input, src):

        x = self.conv1(input)
        x = self.gelu1(x)
        x = self.norm1(x, src)

        x = x + input

        x = self.conv2(x)
        x = self.gelu2(x)
        x = self.norm2(x, src)

        return x

class ConvEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, ndf=64, kw=3):
        super().__init__()

        pw = int(np.ceil((kw - 1.0) / 2))
        
        self.layer1 = nn.Conv2d(1, ndf, kw, stride=2, padding=pw)
        self.norm1  = nn.InstanceNorm2d(ndf, affine=False)
        self.layer2 = nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw)
        self.norm2  = nn.InstanceNorm2d(ndf*2, affine=False)
        self.layer3 = nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw)
        self.norm3  = nn.InstanceNorm2d(ndf*4, affine=False)
        self.layer4 = nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw)
        self.norm4  = nn.InstanceNorm2d(ndf*8, affine=False)
        self.layer5 = nn.Conv2d(ndf*8, ndf*8, kw, stride=2, padding=pw)
        self.norm5  = nn.InstanceNorm2d(ndf*8, affine=False)

        self.so = s0 = 2
        self.fc_mu  = nn.Linear(ndf*8*s0*s0, 64)
        self.fc_var = nn.Linear(ndf*8*s0*s0, 64)

        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, x):
        x = self.norm1(self.layer1(x))
        x = self.norm2(self.layer2(self.actvn(x)))
        x = self.norm3(self.layer3(self.actvn(x)))
        x = self.norm4(self.layer4(self.actvn(x)))
        x = self.norm5(self.layer5(self.actvn(x)))
        
        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        
        mu     = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

class ConvMixer_Generator(nn.Module):
    def __init__(self, dim=256, kernel_size=9):
        super(ConvMixer_Generator, self).__init__()

        # optional Encoder
        self.img_encoder = ConvEncoder()

        # noise embedding
        self.noise_embed = nn.Linear(64, 256*64*64)

        # convmixer block depth = 8
        # self.mixer_block1 = nn.Sequential( Residual(SPADE_ConvMixer_Block(in_ch=dim//2, out_ch=dim//2, kernel_size=kernel_size, group=dim//2, padding="same")), SPADE_ConvMixer_Block(in_ch=dim//2, out_ch=dim//2, kernel_size=1) )
        self.mixer_block1 = SPADE_ConvMixer_Block(dim=dim, kernel_size=kernel_size)
        self.mixer_block2 = SPADE_ConvMixer_Block(dim=dim, kernel_size=kernel_size)
        self.mixer_block3 = SPADE_ConvMixer_Block(dim=dim, kernel_size=kernel_size)
        self.mixer_block4 = SPADE_ConvMixer_Block(dim=dim, kernel_size=kernel_size)
        self.mixer_block5 = SPADE_ConvMixer_Block(dim=dim, kernel_size=kernel_size)
        self.mixer_block6 = SPADE_ConvMixer_Block(dim=dim, kernel_size=kernel_size)
        self.mixer_block7 = SPADE_ConvMixer_Block(dim=dim, kernel_size=kernel_size)
        self.mixer_block8 = SPADE_ConvMixer_Block(dim=dim, kernel_size=kernel_size)

        self.head     = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu     = nn.ReLU()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def forward(self, input):        
        
        mu, logvar = self.img_encoder(input)
        z = self.reparameterize(mu, logvar)
        # z = torch.randn((input.size(0), 64), device='cuda')

        x0 = self.noise_embed(z)          # x0 == [1, 768, 128, 128]
        x0 = x0.view(-1, 256, 64, 64)

        x1 = self.mixer_block1(x0, input)        # x1 == [1, 768, 128, 128]
        x2 = self.mixer_block2(x1, input)        # x2 == (B, 768*2, 128, 128)   
        x3 = self.mixer_block3(x2, input)  
        x4 = self.mixer_block4(x3, input)  
        x5 = self.mixer_block5(x4, input)  
        x6 = self.mixer_block6(x5, input)  
        x7 = self.mixer_block7(x6, input)  
        x8 = self.mixer_block8(x7, input)        # x8 == (B, 768, 128, 128)   

        output = self.head(x8)                # x8 == [1, 1, 512, 512]

        return mu, logvar, self.relu(output+input)

    def inference(self, input):        
        
        mu, logvar = self.img_encoder(input)
        z = self.reparameterize(mu, logvar)

        x0 = self.noise_embed(z)          # x0 == [1, 768, 128, 128]
        x0 = x0.view(-1, 256, 64, 64)

        x1 = self.mixer_block1(x0, input)        # x1 == [1, 768, 128, 128]
        x2 = self.mixer_block2(x1, input)        # x2 == (B, 768*2, 128, 128)   
        x3 = self.mixer_block3(x2, input)  
        x4 = self.mixer_block4(x3, input)  
        x5 = self.mixer_block5(x4, input)  
        x6 = self.mixer_block6(x5, input)  
        x7 = self.mixer_block7(x6, input)  
        x8 = self.mixer_block8(x7, input)        # x8 == (B, 768, 128, 128)   

        output = self.head(x8)                # x8 == [1, 1, 512, 512]

        return self.relu(output+input)


############################################################################################################
# CNN - Based
############################################################################################################
# 1. CMT Unet




############################################################################################################
# GAN - Based
############################################################################################################

# 1. FDGAN

########################################################################################
# Ours
########################################################################################

# 1. FSGAN - base code
class FSGAN(nn.Module):
    def __init__(self):
        super(FSGAN, self).__init__()
        self.Generator          = ConvMixer_Generator(dim=256, kernel_size=9)
        
        self.Low_discriminator  = Low_UNet(in_channels=2+1, repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False)
        self.High_discriminator = High_UNet(in_channels=20+1, repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False)
        
        self.gan_metric         = ls_gan
        self.KLDLoss            = KLDLoss()

    # ref : https://github.com/basiclab/gngan-pytorch
    def normalize_gradient_enc_dec(self, net_D, x, **kwargs):
        """
                        f
        f_hat = --------------------
                || grad_f || + | f |

        reference : https://github.com/basiclab/GNGAN-PyTorch
        """
        x.requires_grad_(True)
        f_enc, f_dec  = net_D(DiffAugment(x, policy='color,translation,cutout'), **kwargs)

        # encoder
        enc_grad      = torch.autograd.grad(f_enc, [x], torch.ones_like(f_enc), create_graph=True, retain_graph=True)[0]
        enc_grad_norm = torch.norm(torch.flatten(enc_grad, start_dim=1), p=2, dim=1)
        enc_grad_norm = enc_grad_norm.view(-1, *[1 for _ in range(len(f_enc.shape) - 1)])
        enc_f_hat     = (f_enc / (enc_grad_norm + torch.abs(f_enc)))

        # decoder
        dec_grad      = torch.autograd.grad(f_dec, [x], torch.ones_like(f_dec), create_graph=True, retain_graph=True)[0]
        dec_grad_norm = torch.norm(torch.flatten(dec_grad, start_dim=1), p=2, dim=1)
        dec_grad_norm = dec_grad_norm.view(-1, *[1 for _ in range(len(f_dec.shape) - 1)])
        dec_f_hat     = (f_dec / (dec_grad_norm + torch.abs(f_dec)))

        return enc_f_hat, dec_f_hat

    def train_Low_Discriminator(self, full_dose, low_dose, gen_full_dose, prefix='Low_Freq', n_iter=0):
        ############## Train Discriminator ###################
        low_real_enc,   low_real_dec     = self.normalize_gradient_enc_dec(self.Low_discriminator, full_dose)
        low_fake_enc,   low_fake_dec     = self.normalize_gradient_enc_dec(self.Low_discriminator, gen_full_dose.detach())
        low_source_enc, low_source_dec   = self.normalize_gradient_enc_dec(self.Low_discriminator, low_dose)

        disc_loss = self.gan_metric(low_real_enc, 1.) + self.gan_metric(low_real_dec, 1.) + \
                    self.gan_metric(low_fake_enc, 0.) + self.gan_metric(low_fake_dec, 0.) + \
                    self.gan_metric(low_source_enc, 0.) + self.gan_metric(low_source_dec, 0.)

        return disc_loss

    def train_High_Discriminator(self, full_dose, low_dose, gen_full_dose, prefix='High_Freq', n_iter=0):
        ############## Train Discriminator ###################
        high_real_enc,   high_real_dec    = self.normalize_gradient_enc_dec(self.High_discriminator, full_dose)
        high_fake_enc,   high_fake_dec    = self.normalize_gradient_enc_dec(self.High_discriminator, gen_full_dose.detach())
        high_source_enc, high_source_dec  = self.normalize_gradient_enc_dec(self.High_discriminator, low_dose)

        disc_loss = self.gan_metric(high_real_enc, 1.) + self.gan_metric(high_real_dec, 1.) + \
                    self.gan_metric(high_fake_enc, 0.) + self.gan_metric(high_fake_dec, 0.) + \
                    self.gan_metric(high_source_enc, 0.) + self.gan_metric(high_source_dec, 0.)

        return disc_loss

        