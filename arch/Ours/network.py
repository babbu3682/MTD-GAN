from re import X
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .DiffAugment_pytorch import DiffAugment
from .Unet_Factory import *
from .DUGAN_wrapper import *




# LOSS
def ls_gan(inputs, targets):
    return torch.mean((inputs - targets) ** 2)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,1,1)  # 1 -> gray channel
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class MSFRLoss(nn.Module):
    def __init__(self):
        super(MSFRLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):

        x_fft = torch.fft.rfftn(x)
        y_fft = torch.fft.rfftn(y)
      
        loss = self.l1_loss(x_fft, y_fft)
        
        return loss



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

class SPADE_Half(nn.Module):
    def __init__(self, norm_type, norm_nc, label_nc):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc//2, affine=False)
        elif norm_type == 'syncbatch':
            self.param_free_norm = nn.SyncBatchNorm(norm_nc//2, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc//2, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' %norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma  = nn.Conv2d(nhidden, norm_nc//2, kernel_size=ks, padding=pw)
        self.mlp_beta   = nn.Conv2d(nhidden, norm_nc//2, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        
        x_norm, x_idt = torch.chunk(x, 2, dim=1)
        
        # Part 1. generate parameter-free normalized activations
        x_norm = self.param_free_norm(x_norm)
        
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bicubic')
        actv   = self.mlp_shared(segmap)
        gamma  = self.mlp_gamma(actv)
        beta   = self.mlp_beta(actv)

        # apply scale and bias
        x_norm = x_norm*gamma + beta

        out = torch.cat([x_norm, x_idt], dim=1)

        return out


# Generator
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=None, relu=None):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size//2, stride=stride, bias=bias)
        self.norm_type = norm

        # Norm
        if norm == 'half':
            self.norm = nn.InstanceNorm2d(out_channel//2, affine=True)
        elif norm == 'spade':
            self.norm       = nn.InstanceNorm2d(out_channel, affine=False)
            self.mlp_shared = nn.Sequential(nn.Conv2d(1, 128, kernel_size=3, padding=3//2), nn.ReLU())
            self.mlp_gamma  = nn.Conv2d(128, out_channel, kernel_size=3, padding=3//2)
            self.mlp_beta   = nn.Conv2d(128, out_channel, kernel_size=3, padding=3//2)
        else: 
            self.norm = None

        # Act
        if relu == 'relu':
            self.relu = nn.ReLU(inplace=True)
        else :
            self.relu = None

    def forward(self, x, src=None):
        x = self.conv(x)

        if self.norm is not None:
            if self.norm_type == 'half':
                x_norm, x_idt = torch.chunk(x, 2, dim=1)
                x_norm        = self.norm(x_norm)
                x             = torch.cat([x_norm, x_idt], dim=1)
            elif self.norm_type == 'spade':
                x_norm = self.norm(x)
                segmap = F.interpolate(src, size=x_norm.size()[2:], mode='bicubic')
                actv   = self.mlp_shared(segmap)
                gamma  = self.mlp_gamma(actv)
                beta   = self.mlp_beta(actv)
                # Apply scale & bias
                x = x_norm*gamma + beta
            else :
                x = self.norm(x)

        if self.relu is not None:
            x = self.relu(x)
        
        return x

class Encode_Block(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(Encode_Block, self).__init__()
        # Stem
        self.conv  = Conv_Block(in_channels, out_channels, kernel_size=3, stride=1, norm='none', relu='relu')

        # Fourier domain        
        self.fft_conv1 = Conv_Block(out_channels*2, out_channels*2, kernel_size=1, stride=1,  norm='none', relu='relu')
        self.fft_conv2 = Conv_Block(out_channels*2, out_channels*2, kernel_size=1, stride=1,  norm='none', relu='none')

        # Image domain
        self.img_conv1 = Conv_Block(out_channels, out_channels, kernel_size=3, stride=1, norm='half', relu='relu')
        self.img_conv2 = Conv_Block(out_channels, out_channels, kernel_size=3, stride=1, norm='none', relu='none')
        
        # Mixing
        self.conv1x1 = Conv_Block(out_channels*3, out_channels, kernel_size=1, stride=1, bias=True, norm='none', relu='relu')

    def forward(self, x):
        # Stem
        x = self.conv(x)
        _, _, H, W = x.shape

        # Fourier domain   
        fft = torch.fft.rfft2(x, norm='backward')
        fft = torch.cat([fft.real, fft.imag], dim=1)
        fft = self.fft_conv2(self.fft_conv1(fft))        
        fft_real, fft_imag = torch.chunk(fft, 2, dim=1)        
        fft = torch.complex(fft_real, fft_imag)
        fft = torch.fft.irfft2(fft, s=(H, W), norm='backward')
        
        # Image domain  
        img = self.img_conv1(x)
        img = self.img_conv2(img)

        # Mixing
        mix    = torch.cat([x, img, fft], dim=1)
        output = self.conv1x1(mix)

        return output

class Decode_Block(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(Decode_Block, self).__init__()

        # Stem
        self.conv  = Conv_Block(in_channels, out_channels, kernel_size=3, stride=1, norm='none', relu='none')

        # Fourier domain        
        self.fft_conv1 = Conv_Block(out_channels*2, out_channels*2, kernel_size=1, stride=1,  norm='none', relu='relu')
        self.fft_conv2 = Conv_Block(out_channels*2, out_channels*2, kernel_size=1, stride=1,  norm='none', relu='none')

        # Image domain
        self.img_conv1 = Conv_Block(out_channels, out_channels, kernel_size=3, stride=1, norm='spade', relu='relu')
        self.img_conv2 = Conv_Block(out_channels, out_channels, kernel_size=3, stride=1, norm='spade', relu='none')
        
        # Mixing
        self.conv1x1 = Conv_Block(out_channels*3, out_channels, kernel_size=1, stride=1, bias=True, norm='none', relu='relu')


    def forward(self, x, src):
        # Stem
        x = self.conv(x)
        _, _, H, W = x.shape

        # Fourier domain   
        fft = torch.fft.rfft2(x, norm='backward')
        fft = torch.cat([fft.real, fft.imag], dim=1)
        fft = self.fft_conv2(self.fft_conv1(fft))        
        fft_real, fft_imag = torch.chunk(fft, 2, dim=1)        
        fft = torch.complex(fft_real, fft_imag)
        fft = torch.fft.irfft2(fft, s=(H, W), norm='backward')
        
        # Image domain  
        img = self.img_conv1(x, src)
        img = self.img_conv2(img, src)

        # Mixing
        mix    = torch.cat([x, img, fft], dim=1)
        output = self.conv1x1(mix)

        return output

class Bottle_Block(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(Bottle_Block, self).__init__()
        # Stem
        self.conv  = Conv_Block(in_channels, out_channels, kernel_size=3, stride=1, norm='none', relu='none')

        # Fourier domain        
        self.fft_conv1 = Conv_Block(out_channels*2, out_channels*2, kernel_size=1, stride=1,  norm='none', relu='relu')
        self.fft_conv2 = Conv_Block(out_channels*2, out_channels*2, kernel_size=1, stride=1,  norm='none', relu='none')

        # Image domain
        self.img_conv1 = Conv_Block(out_channels, out_channels, kernel_size=3, stride=1, norm='half', relu='relu')
        self.img_conv2 = Conv_Block(out_channels, out_channels, kernel_size=3, stride=1, norm='none', relu='none')
        
        # Mixing
        self.conv1x1 = Conv_Block(out_channels*3, out_channels, kernel_size=1, stride=1, bias=True, norm='none', relu='relu')

        # Attention
        self.attn    = AttentionBlockA(dim_in=out_channels, dim_out=out_channels//4, heads=8)

    def forward(self, x):
        # Stem
        x = self.conv(x)
        _, _, H, W = x.shape

        # Fourier domain   
        fft = torch.fft.rfft2(x, norm='backward')
        fft = torch.cat([fft.real, fft.imag], dim=1)
        fft = self.fft_conv2(self.fft_conv1(fft))        
        fft_real, fft_imag = torch.chunk(fft, 2, dim=1)        
        fft = torch.complex(fft_real, fft_imag)
        fft = torch.fft.irfft2(fft, s=(H, W), norm='backward')
        
        # Image domain  
        img = self.img_conv1(x)
        img = self.img_conv2(img)

        # Mixing
        mix    = torch.cat([x, img, fft], dim=1)
        output = self.conv1x1(mix)
        
        # Attention
        output = self.attn(output)

        return output

class ResFFT_Generator(nn.Module):
    def __init__(self, channel=64):
        super(ResFFT_Generator, self).__init__()

        self.enc_layer1   = nn.Sequential(*nn.ModuleList([Encode_Block(in_channels=1, out_channels=channel*2)]+[Encode_Block(in_channels=channel*2, out_channels=channel*2) for _ in range(7)]))
        
        self.pool2        = Downsample_Unet(n_feat=128)
        self.enc_layer2   = nn.Sequential(*nn.ModuleList([Encode_Block(in_channels=channel*2, out_channels=channel*4)]+[Encode_Block(in_channels=channel*4, out_channels=channel*4) for _ in range(7)]))
        
        self.pool3        = Downsample_Unet(n_feat=256)        
        self.bot3         = nn.Sequential(*nn.ModuleList([Bottle_Block(in_channels=channel*4, out_channels=channel*4)]+[Bottle_Block(in_channels=channel*4, out_channels=channel*4) for _ in range(7)]))
        self.unpool3      = Upsample_Unet(n_feat=256)

        self.dec_layer1   = nn.ModuleList([Decode_Block(in_channels=channel*4, out_channels=channel*2)]+[Decode_Block(in_channels=channel*2, out_channels=channel*2) for _ in range(7)])
        self.unpool4      = Upsample_Unet(n_feat=128)
        
        self.dec_layer2   = nn.ModuleList([Decode_Block(in_channels=channel*2, out_channels=channel)]+[Decode_Block(in_channels=channel, out_channels=channel) for _ in range(7)])

        # Head                    
        self.fc           = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu         = nn.ReLU()

    def two_input_list(self, x, input, layers):
        for block in layers:
            x = block(x, input)
        return x        

    def forward(self, input):
        x = self.enc_layer1(input)     
        x = self.pool2(x)
        x = self.enc_layer2(x) 
        
        x = self.pool3(x)
        x = self.bot3(x)
        x = self.unpool3(x)

        x = self.two_input_list(x, input, self.dec_layer1)
        x = self.unpool4(x)

        x = self.two_input_list(x, input, self.dec_layer2)

        # Head
        x = self.fc(x)
        x = self.relu(x)

        return torch.clamp(x, min=0.0, max=1.0)










############################################################################################################
# GAN - Based
############################################################################################################
# 1. FD-GAN
def turn_on_spectral_norm(module):
    module_output = module
    if isinstance(module, torch.nn.Conv2d):
        if module.out_channels != 1 and module.in_channels > 4:
            module_output = nn.utils.spectral_norm(module)
    # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #     module_output = nn.utils.spectral_norm(module)
    for name, child in module.named_children():
        module_output.add_module(name, turn_on_spectral_norm(child))
    del module
    return module_output

class FFT_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFT_Conv2d, self).__init__()

        self.conv   = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels*2, out_channels*2, kernel_size=1, stride=1))
        self.relu   = nn.LeakyReLU(0.2)

    def forward(self, x):
        _, _, H, W = x.shape

        x   = torch.fft.rfft2(x, norm='backward')
        x_f = torch.cat([x.real, x.imag], dim=1)
        x_f = self.relu(self.conv(x_f))
        
        x_real, x_imag = torch.chunk(x_f, 2, dim=1)
        x   = torch.complex(x_real, x_imag)
        x   = torch.fft.irfft2(x, s=(H, W), norm='backward')

        return x

class Image_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels*5, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.brelu    = nn.LeakyReLU(0.2)                
        self.bconv    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.brelu    = nn.LeakyReLU(0.2)                

        # Dec
        self.up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu11   = nn.LeakyReLU(0.2)                
        self.dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu12   = nn.LeakyReLU(0.2)        

        self.up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu21   = nn.LeakyReLU(0.2)                
        self.dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu22   = nn.LeakyReLU(0.2)        

        self.up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.drelu31   = nn.LeakyReLU(0.2)                
        self.dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.drelu32   = nn.LeakyReLU(0.2)        

        self.up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.drelu41   = nn.LeakyReLU(0.2)                
        self.dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.drelu42   = nn.LeakyReLU(0.2)        

        self.up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.drelu51   = nn.LeakyReLU(0.2)                
        self.dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.drelu52   = nn.LeakyReLU(0.2)    

        self.up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.drelu61   = nn.LeakyReLU(0.2)                
        self.dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.drelu62   = nn.LeakyReLU(0.2)    

        # Head
        self.enc_out   = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.3, inplace=True), nn.Linear(512, 1))
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        input = self.window_filter(input)
        
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv51(x))
        x6 = self.relu62(self.conv52(x))
        x = self.down6(x6)
        
        # Bot
        x = self.brelu(self.bconv(x))
        x_bot = self.brelu(self.bconv(x))                
        
        # Decoder
        x = self.up1(x_bot)
        x = self.drelu11(self.dconv11(torch.cat([x, x6], dim=1)))
        x = self.drelu12(self.dconv12(x))                
        

        x = self.up2(x)
        x = self.drelu21(self.dconv21(torch.cat([x, x5], dim=1)))
        x = self.drelu22(self.dconv22(x))              
        

        x = self.up3(x)
        x = self.drelu31(self.dconv31(torch.cat([x, x4], dim=1)))
        x = self.drelu32(self.dconv32(x))              
        

        x = self.up4(x)
        x = self.drelu41(self.dconv41(torch.cat([x, x3], dim=1)))
        x = self.drelu42(self.dconv42(x))              
        

        x = self.up5(x)
        x = self.drelu51(self.dconv51(torch.cat([x, x2], dim=1)))
        x = self.drelu52(self.dconv52(x))              
        

        x = self.up6(x)
        x = self.drelu61(self.dconv61(torch.cat([x, x1], dim=1)))
        x = self.drelu62(self.dconv62(x))              

        x_enc = self.enc_out(x_bot) 
        x_dec = self.dec_out(x)

        return x_enc, x_dec

    def window_filter(self, x):
        weight             = torch.ones(size=(5, 1, 1, 1)).cuda()      
        weight[0, :, :, :] = 50.0
        weight[1, :, :, :] = 31.250
        weight[2, :, :, :] = 45.455
        weight[3, :, :, :] = 1.464
        weight[4, :, :, :] = 11.628
        bias    = torch.ones(size=(5,)).cuda()        
        bias[0] =  -12.5
        bias[1] =  -7.687
        bias[2] =  -11.682
        bias[3] =  -0.081
        bias[4] =  -2.465        

        x = F.conv2d(x, weight, bias)
        x = torch.minimum(torch.maximum(x, torch.tensor(0)), torch.tensor(1.0))
        return x

class Fourier_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = FFT_Conv2d(in_channels, out_channels)
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = FFT_Conv2d(out_channels, out_channels)
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = FFT_Conv2d(out_channels, out_channels*2)
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = FFT_Conv2d(out_channels*2, out_channels*2)
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = FFT_Conv2d(out_channels*2, out_channels*4)
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = FFT_Conv2d(out_channels*4, out_channels*4)
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = FFT_Conv2d(out_channels*4, out_channels*8)
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = FFT_Conv2d(out_channels*8, out_channels*8)
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = FFT_Conv2d(out_channels*8, out_channels*8)
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = FFT_Conv2d(out_channels*8, out_channels*8)
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = FFT_Conv2d(out_channels*8, out_channels*8)
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = FFT_Conv2d(out_channels*8, out_channels*8)
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv    = FFT_Conv2d(out_channels*8, out_channels*8)
        self.brelu    = nn.LeakyReLU(0.2)                
        self.bconv    = FFT_Conv2d(out_channels*8, out_channels*8)
        self.brelu    = nn.LeakyReLU(0.2)                

        # Dec
        self.up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv11   = FFT_Conv2d(out_channels*8*2, out_channels*8)
        self.drelu11   = nn.LeakyReLU(0.2)                
        self.dconv12   = FFT_Conv2d(out_channels*8, out_channels*8)
        self.drelu12   = nn.LeakyReLU(0.2)        

        self.up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv21   = FFT_Conv2d(out_channels*8*2, out_channels*8)
        self.drelu21   = nn.LeakyReLU(0.2)                
        self.dconv22   = FFT_Conv2d(out_channels*8, out_channels*8)
        self.drelu22   = nn.LeakyReLU(0.2)        

        self.up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv31   = FFT_Conv2d(out_channels*8*2, out_channels*4)
        self.drelu31   = nn.LeakyReLU(0.2)                
        self.dconv32   = FFT_Conv2d(out_channels*4, out_channels*4)
        self.drelu32   = nn.LeakyReLU(0.2)        

        self.up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv41   = FFT_Conv2d(out_channels*4*2, out_channels*2)
        self.drelu41   = nn.LeakyReLU(0.2)                
        self.dconv42   = FFT_Conv2d(out_channels*2, out_channels*2)
        self.drelu42   = nn.LeakyReLU(0.2)        

        self.up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv51   = FFT_Conv2d(out_channels*2*2, out_channels)
        self.drelu51   = nn.LeakyReLU(0.2)                
        self.dconv52   = FFT_Conv2d(out_channels, out_channels)
        self.drelu52   = nn.LeakyReLU(0.2)    

        self.up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv61   = FFT_Conv2d(out_channels*2, 1)
        self.drelu61   = nn.LeakyReLU(0.2)                
        self.dconv62   = FFT_Conv2d(1, 1)
        self.drelu62   = nn.LeakyReLU(0.2)    

        # Head
        self.enc_out   = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.3, inplace=True), nn.Linear(512, 1))
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # input = self.window_filter(input)
        
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv51(x))
        x6 = self.relu62(self.conv52(x))
        x = self.down6(x6)
        
        # Bot
        x = self.brelu(self.bconv(x))
        x_bot = self.brelu(self.bconv(x))                
        
        # Decoder
        x = self.up1(x_bot)
        x = self.drelu11(self.dconv11(torch.cat([x, x6], dim=1)))
        x = self.drelu12(self.dconv12(x))                
        

        x = self.up2(x)
        x = self.drelu21(self.dconv21(torch.cat([x, x5], dim=1)))
        x = self.drelu22(self.dconv22(x))              
        

        x = self.up3(x)
        x = self.drelu31(self.dconv31(torch.cat([x, x4], dim=1)))
        x = self.drelu32(self.dconv32(x))              
        

        x = self.up4(x)
        x = self.drelu41(self.dconv41(torch.cat([x, x3], dim=1)))
        x = self.drelu42(self.dconv42(x))              
        

        x = self.up5(x)
        x = self.drelu51(self.dconv51(torch.cat([x, x2], dim=1)))
        x = self.drelu52(self.dconv52(x))              
        

        x = self.up6(x)
        x = self.drelu61(self.dconv61(torch.cat([x, x1], dim=1)))
        x = self.drelu62(self.dconv62(x))              

        # Head
        x_enc = self.enc_out(x_bot) 
        x_dec = self.dec_out(x)

        return x_enc, x_dec

    def window_filter(self, x):
        weight             = torch.ones(size=(5, 1, 1, 1)).cuda()      
        weight[0, :, :, :] = 50.0
        weight[1, :, :, :] = 31.250
        weight[2, :, :, :] = 45.455
        weight[3, :, :, :] = 1.464
        weight[4, :, :, :] = 11.628
        bias    = torch.ones(size=(5,)).cuda()        
        bias[0] =  -12.5
        bias[1] =  -7.687
        bias[2] =  -11.682
        bias[3] =  -0.081
        bias[4] =  -2.465        

        x = F.conv2d(x, weight, bias)
        x = torch.minimum(torch.maximum(x, torch.tensor(0)), torch.tensor(1.0))
        return x

class FDGAN(nn.Module):
    def __init__(self):
        super(FDGAN, self).__init__()
        # Generator
        self.Generator             = ResFFT_Generator()

        # Discriminator
        self.Image_Discriminator   = Image_UNet(in_channels=1, out_channels=64)
        self.Fourier_Discriminator = Fourier_UNet(in_channels=1, out_channels=64)

        # LOSS
        # self.gan_metric    = nn.BCEWithLogitsLoss()
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()
        self.fourier_loss   = MSFRLoss()        
        self.edge_loss      = EdgeLoss()

    def Image_d_loss(self, x, y):
        fake                   = self.Generator(x).detach()   
        real_enc,  real_dec    = self.Image_Discriminator(DiffAugment(y))
        fake_enc,  fake_dec    = self.Image_Discriminator(DiffAugment(fake))
        input_enc, input_dec   = self.Image_Discriminator(DiffAugment(x))
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + \
                    self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.) + \
                    self.gan_metric(input_enc, 0.) + self.gan_metric(input_dec, 0.)

        return disc_loss


    def Fourier_d_loss(self, x, y):
        fourier_fake                           = self.Generator(x).detach()  
        fourier_real_enc,   fourier_real_dec   = self.Fourier_Discriminator(DiffAugment(y))
        fourier_fake_enc,   fourier_fake_dec   = self.Fourier_Discriminator(DiffAugment(fourier_fake))
        fourier_input_enc,  fourier_input_dec  = self.Fourier_Discriminator(DiffAugment(x))

        disc_loss = self.gan_metric(fourier_real_enc, 1.) + self.gan_metric(fourier_real_dec, 1.) + \
                    self.gan_metric(fourier_fake_enc, 0.) + self.gan_metric(fourier_fake_dec, 0.) + \
                    self.gan_metric(fourier_input_enc, 0.) + self.gan_metric(fourier_input_dec, 0.)
        
        return disc_loss


    def g_loss(self, x, y):
        fake                              = self.Generator(x)
        image_gen_enc,   image_gen_dec    = self.Image_Discriminator(fake)
        fourier_gen_enc, fourier_gen_dec  = self.Fourier_Discriminator(fake)
        
        image_gen_loss                = self.gan_metric(image_gen_enc, 1.) + self.gan_metric(image_gen_dec, 1.)
        fourier_gen_loss              = self.gan_metric(fourier_gen_enc, 1.) + self.gan_metric(fourier_gen_dec, 1.)

        adv_loss     = image_gen_loss + fourier_gen_loss 
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        fourier_loss = 50.0*self.fourier_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        total_loss = adv_loss + pix_loss + fourier_loss + edge_loss

        return total_loss

# 2. FD-GAN-Domain
class Fourier_domain_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels*5*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.brelu    = nn.LeakyReLU(0.2)                
        self.bconv    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.brelu    = nn.LeakyReLU(0.2)                

        # Dec
        self.up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu11   = nn.LeakyReLU(0.2)                
        self.dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu12   = nn.LeakyReLU(0.2)        

        self.up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu21   = nn.LeakyReLU(0.2)                
        self.dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu22   = nn.LeakyReLU(0.2)        

        self.up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.drelu31   = nn.LeakyReLU(0.2)                
        self.dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.drelu32   = nn.LeakyReLU(0.2)        

        self.up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.drelu41   = nn.LeakyReLU(0.2)                
        self.dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.drelu42   = nn.LeakyReLU(0.2)        

        self.up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.drelu51   = nn.LeakyReLU(0.2)                
        self.dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.drelu52   = nn.LeakyReLU(0.2)    

        self.up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.drelu61   = nn.LeakyReLU(0.2)                
        self.dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.drelu62   = nn.LeakyReLU(0.2)    

        # Head
        self.enc_out   = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.3, inplace=True), nn.Linear(512, 1))
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        input  = self.window_filter(input)
        input  = torch.fft.fft2(input, norm='backward')
        input  = torch.fft.fftshift(input)
        input  = torch.cat([input.real, input.imag], dim=1)

        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv51(x))
        x6 = self.relu62(self.conv52(x))
        x = self.down6(x6)
        
        # Bot
        x = self.brelu(self.bconv(x))
        x_bot = self.brelu(self.bconv(x))                
        
        # Decoder
        x = self.up1(x_bot)
        x = self.drelu11(self.dconv11(torch.cat([x, x6], dim=1)))
        x = self.drelu12(self.dconv12(x))                
        

        x = self.up2(x)
        x = self.drelu21(self.dconv21(torch.cat([x, x5], dim=1)))
        x = self.drelu22(self.dconv22(x))              
        

        x = self.up3(x)
        x = self.drelu31(self.dconv31(torch.cat([x, x4], dim=1)))
        x = self.drelu32(self.dconv32(x))              
        

        x = self.up4(x)
        x = self.drelu41(self.dconv41(torch.cat([x, x3], dim=1)))
        x = self.drelu42(self.dconv42(x))              
        

        x = self.up5(x)
        x = self.drelu51(self.dconv51(torch.cat([x, x2], dim=1)))
        x = self.drelu52(self.dconv52(x))              
        

        x = self.up6(x)
        x = self.drelu61(self.dconv61(torch.cat([x, x1], dim=1)))
        x = self.drelu62(self.dconv62(x))              

        x_enc = self.enc_out(x_bot) 
        x_dec = self.dec_out(x)

        return x_enc, x_dec

    def window_filter(self, x):
        weight             = torch.ones(size=(5, 1, 1, 1)).cuda()      
        weight[0, :, :, :] = 50.0
        weight[1, :, :, :] = 31.250
        weight[2, :, :, :] = 45.455
        weight[3, :, :, :] = 1.464
        weight[4, :, :, :] = 11.628
        bias    = torch.ones(size=(5,)).cuda()        
        bias[0] =  -12.5
        bias[1] =  -7.687
        bias[2] =  -11.682
        bias[3] =  -0.081
        bias[4] =  -2.465        

        x = F.conv2d(x, weight, bias)
        x = torch.minimum(torch.maximum(x, torch.tensor(0)), torch.tensor(1.0))
        return x

class FDGAN_domain(nn.Module):
    def __init__(self):
        super(FDGAN_domain, self).__init__()
        # Generator
        self.Generator             = ResFFT_Generator()

        # Discriminator
        self.Image_Discriminator   = Image_UNet(in_channels=1, out_channels=64)
        self.Fourier_Discriminator = Fourier_domain_UNet(in_channels=1, out_channels=64)

        # LOSS
        # self.gan_metric    = nn.BCEWithLogitsLoss()
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()
        self.fourier_loss   = MSFRLoss()        
        self.edge_loss      = EdgeLoss()

    def Image_d_loss(self, x, y):
        fake                   = self.Generator(x).detach()   
        real_enc,  real_dec    = self.Image_Discriminator(DiffAugment(y))
        fake_enc,  fake_dec    = self.Image_Discriminator(DiffAugment(fake))
        input_enc, input_dec   = self.Image_Discriminator(DiffAugment(x))
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + \
                    self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.) + \
                    self.gan_metric(input_enc, 0.) + self.gan_metric(input_dec, 0.)

        return disc_loss


    def Fourier_d_loss(self, x, y):
        fourier_fake                           = self.Generator(x).detach() 
        fourier_real_enc,   fourier_real_dec   = self.Fourier_Discriminator(y)
        fourier_fake_enc,   fourier_fake_dec   = self.Fourier_Discriminator(fourier_fake)
        fourier_input_enc,  fourier_input_dec  = self.Fourier_Discriminator(x)

        disc_loss = self.gan_metric(fourier_real_enc, 1.) + self.gan_metric(fourier_real_dec, 1.) + \
                    self.gan_metric(fourier_fake_enc, 0.) + self.gan_metric(fourier_fake_dec, 0.) + \
                    self.gan_metric(fourier_input_enc, 0.) + self.gan_metric(fourier_input_dec, 0.)
        
        return disc_loss


    def g_loss(self, x, y):
        fake                              = self.Generator(x)
        image_gen_enc,   image_gen_dec    = self.Image_Discriminator(fake)
        fourier_gen_enc, fourier_gen_dec  = self.Fourier_Discriminator(fake)
        
        image_gen_loss                = self.gan_metric(image_gen_enc, 1.) + self.gan_metric(image_gen_dec, 1.)
        fourier_gen_loss              = self.gan_metric(fourier_gen_enc, 1.) + self.gan_metric(fourier_gen_dec, 1.)

        adv_loss     = image_gen_loss + fourier_gen_loss 
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        fourier_loss = 50.0*self.fourier_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        total_loss = adv_loss + pix_loss + fourier_loss + edge_loss

        return total_loss

    

# 3. MTL_D_GAN - Real Ours
# Reference 
# https://github.com/pkumivision/FFC/issues/3
# https://github.com/INVOKERer/DeepRFT/blob/06f109e1c959cc7b3ea855d5d5feecf4d88f09a8/layers.py#L92

def minmax_normalize(x):
    if len(torch.unique(x)) != 1:  # Sometimes it cause the nan inputs...
        x -= x.min()
        x /= x.max() 
    return x

def ls_gan_with_weight(inputs, targets, diffs):
    return torch.mean( minmax_normalize(torch.abs(diffs)) * (inputs - targets)**2 )

def ls_gan_with_weight_upgrade(inputs, targets, diffs):
    return torch.mean( torch.abs(diffs).bool() * (inputs - targets)**2 )


class REDCNN_Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=96, num_layers=10, kernel_size=5, padding=0):
        super(REDCNN_Generator, self).__init__()
        encoder = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        decoder = [nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        for _ in range(num_layers):
            encoder.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            decoder.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, x: torch.Tensor):
        residuals = []
        for block in self.encoder:
            residuals.append(x)
            x = F.relu(block(x))
        for residual, block in zip(residuals[::-1], self.decoder[::-1]):
            x = F.relu(block(x) + residual)
        return x

class FFT_ConvBlock(nn.Module):
    def __init__(self, out_channels):
        super(FFT_ConvBlock, self).__init__()
        self.img_conv  = nn.Conv2d(out_channels,   out_channels,   kernel_size=3, stride=1, padding=1)
        self.fft_conv  = nn.Conv2d(out_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Fourier domain   
        _, _, H, W = x.shape
        fft = torch.fft.rfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        fft = torch.cat([fft.real, fft.imag], dim=1)
        fft = F.relu(self.fft_conv(fft))
        fft_real, fft_imag = torch.chunk(fft, 2, dim=1)        
        fft = torch.complex(fft_real, fft_imag)
        fft = torch.fft.irfft2(fft, s=(H, W), dim=(2, 3), norm='ortho')
        
        # Image domain  
        img = F.relu(self.img_conv(x))

        # Mixing (residual, image, fourier)
        output = x + img + fft
        return output        

class FFT_Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=96, num_layers=10, kernel_size=5, padding=0):
        super(FFT_Generator, self).__init__()
        encoder = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        decoder = [nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        enforce = []
        for _ in range(num_layers):
            encoder.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            decoder.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        for _ in range(21):
            enforce.append(FFT_ConvBlock(out_channels))
        self.enforce = nn.ModuleList(enforce)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)


    def forward(self, x: torch.Tensor):
    
        e1 = F.relu(self.encoder[0](x))
        e1 = self.enforce[0](e1)

        e2 = F.relu(self.encoder[1](e1))
        e2 = self.enforce[1](e2)

        e3 = F.relu(self.encoder[2](e2))
        e3 = self.enforce[2](e3)

        e4 = F.relu(self.encoder[3](e3))
        e4 = self.enforce[3](e4)

        e5 = F.relu(self.encoder[4](e4))
        e5 = self.enforce[4](e5)

        e6 = F.relu(self.encoder[5](e5))
        e6 = self.enforce[5](e6)

        e7 = F.relu(self.encoder[6](e6))
        e7 = self.enforce[6](e7)

        e8 = F.relu(self.encoder[7](e7))
        e8 = self.enforce[7](e8)

        e9 = F.relu(self.encoder[8](e8))
        e9 = self.enforce[8](e9)

        e10 = F.relu(self.encoder[9](e9))
        e10 = self.enforce[9](e10)

        # Bottleneck
        x_b = F.relu(self.encoder[10](e10))
        x_b = self.enforce[10](x_b)                

        # Decoder
        d10 = F.relu(self.decoder[-1](x_b) + e10)

        d9  = self.enforce[11](d10)
        d9  = F.relu(self.decoder[-2](d9) + e9)

        d8  = self.enforce[12](d9)
        d8  = F.relu(self.decoder[-3](d8) + e8)

        d7  = self.enforce[13](d8)
        d7  = F.relu(self.decoder[-4](d7) + e7)
          
        d6  = self.enforce[14](d7)
        d6  = F.relu(self.decoder[-5](d6) + e6)

        d5  = self.enforce[15](d6)
        d5  = F.relu(self.decoder[-6](d5) + e5)

        d4  = self.enforce[16](d5)
        d4  = F.relu(self.decoder[-7](d4) + e4)

        d3  = self.enforce[17](d4)
        d3  = F.relu(self.decoder[-8](d3) + e3)

        d2  = self.enforce[18](d3)
        d2  = F.relu(self.decoder[-9](d2) + e2)

        d1  = self.enforce[19](d2)
        d1  = F.relu(self.decoder[-10](d1) + e1)

        d0  = self.enforce[20](d1)
        d0  = F.relu(self.decoder[-11](d0) + x)                                                          
        
        return d0

class UpsampleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels*(scale**2), kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(upscale_factor=scale)
        )

    def forward(self, input):
        return self.upsample(input)

# GeM pool: https://arxiv.org/pdf/1711.02512.pdf
# https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/6d4ce7854198f132176965761a3dc26fffaf66c5
# https://github.com/damo-cv/TransReID-SSL/blob/fc39e88240aa7cb7b28dd2097e7f161ae2be3ad8/transreid_pytorch/model/backbones/vit_pytorch.py
class GeneralizedMeanPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        self.p   = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def gem(self, x, p=3, eps=1e-6):
        return F.adaptive_avg_pool2d(input=x.clamp(min=eps).pow(p), output_size=1).pow(1./p)

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

class Multi_Task_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot --> 이미 input이 1x1이기에 kernel_size를 1로 하는게 더 좋을듯 하다....
        self.bconv    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.brelu    = nn.LeakyReLU(0.2)                
        self.bconv    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.brelu    = nn.LeakyReLU(0.2)                

        # SEG Dec
        self.s_up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu11   = nn.LeakyReLU(0.2)                
        self.s_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu12   = nn.LeakyReLU(0.2)        

        self.s_up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu21   = nn.LeakyReLU(0.2)                
        self.s_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu22   = nn.LeakyReLU(0.2)        

        self.s_up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu31   = nn.LeakyReLU(0.2)                
        self.s_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu32   = nn.LeakyReLU(0.2)        

        self.s_up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu41   = nn.LeakyReLU(0.2)                
        self.s_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu42   = nn.LeakyReLU(0.2)        

        self.s_up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu51   = nn.LeakyReLU(0.2)                
        self.s_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu52   = nn.LeakyReLU(0.2)    

        self.s_up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu61   = nn.LeakyReLU(0.2)                
        self.s_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu62   = nn.LeakyReLU(0.2)    

        # REC Dec
        self.r_up1       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu11   = nn.LeakyReLU(0.2)                
        self.r_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu12   = nn.LeakyReLU(0.2)        

        self.r_up2       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu21   = nn.LeakyReLU(0.2)                
        self.r_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu22   = nn.LeakyReLU(0.2)        

        self.r_up3       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu31   = nn.LeakyReLU(0.2)                
        self.r_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu32   = nn.LeakyReLU(0.2)        

        self.r_up4       = UpsampleBlock(scale=2, input_channels=out_channels*4, output_channels=out_channels*4)
        self.r_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu41   = nn.LeakyReLU(0.2)                
        self.r_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu42   = nn.LeakyReLU(0.2)        

        self.r_up5       = UpsampleBlock(scale=2, input_channels=out_channels*2, output_channels=out_channels*2)
        self.r_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu51   = nn.LeakyReLU(0.2)                
        self.r_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu52   = nn.LeakyReLU(0.2)    

        self.r_up6       = UpsampleBlock(scale=2, input_channels=out_channels, output_channels=out_channels)
        self.r_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu61   = nn.LeakyReLU(0.2)                
        self.r_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu62   = nn.LeakyReLU(0.2)  

        # Heads
        self.enc_out   = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.3), nn.Linear(512, 1))
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        self.rec_out   = nn.Conv2d(in_channels, 1, 1)
        
        # POOL
        self.gem_pool  = GeneralizedMeanPooling()

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv51(x))
        x6 = self.relu62(self.conv52(x))
        x = self.down6(x6)
        
        # Bottleneck
        x = self.brelu(self.bconv(x))
        x_bot = self.brelu(self.bconv(x))                
        
        # SEG Decoder
        x = self.s_up1(x_bot)
        x = self.s_drelu11(self.s_dconv11(torch.cat([x, x6], dim=1)))
        x = self.s_drelu12(self.s_dconv12(x))                

        x = self.s_up2(x)
        x = self.s_drelu21(self.s_dconv21(torch.cat([x, x5], dim=1)))
        x = self.s_drelu22(self.s_dconv22(x))              

        x = self.s_up3(x)
        x = self.s_drelu31(self.s_dconv31(torch.cat([x, x4], dim=1)))
        x = self.s_drelu32(self.s_dconv32(x))              

        x = self.s_up4(x)
        x = self.s_drelu41(self.s_dconv41(torch.cat([x, x3], dim=1)))
        x = self.s_drelu42(self.s_dconv42(x))              

        x = self.s_up5(x)
        x = self.s_drelu51(self.s_dconv51(torch.cat([x, x2], dim=1)))
        x = self.s_drelu52(self.s_dconv52(x))              

        x = self.s_up6(x)
        x = self.s_drelu61(self.s_dconv61(torch.cat([x, x1], dim=1)))
        seg_out = self.s_drelu62(self.s_dconv62(x))              

        # REC Decoder
        x = self.r_up1(x_bot)
        x = self.r_drelu11(self.r_dconv11(x))
        x = self.r_drelu12(self.r_dconv12(x))                

        x = self.r_up2(x)
        x = self.r_drelu21(self.r_dconv21(x))
        x = self.r_drelu22(self.r_dconv22(x))              

        x = self.r_up3(x)
        x = self.r_drelu31(self.r_dconv31(x))
        x = self.r_drelu32(self.r_dconv32(x))              

        x = self.r_up4(x)
        x = self.r_drelu41(self.r_dconv41(x))
        x = self.r_drelu42(self.r_dconv42(x))              

        x = self.r_up5(x)
        x = self.r_drelu51(self.r_dconv51(x))
        x = self.r_drelu52(self.r_dconv52(x))              

        x = self.r_up6(x)
        x = self.r_drelu61(self.r_dconv61(x))
        rec_out = self.r_drelu62(self.r_dconv62(x))       

        # Heads
        x_enc = self.enc_out(x_bot)   # enc에 FC 1 layer 더 있어야 편해질듯...?
        x_dec = self.dec_out(seg_out)
        x_rec = self.rec_out(rec_out)

        return x_enc, x_dec, x_rec

class Multi_Task_Discriminator_Skip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot --> 이미 input이 1x1이기에 kernel_size를 1로 하는게 더 좋을듯 하다....
        self.bconv    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.brelu    = nn.LeakyReLU(0.2)                
        self.bconv    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.brelu    = nn.LeakyReLU(0.2)                

        # SEG Dec
        self.s_up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu11   = nn.LeakyReLU(0.2)                
        self.s_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu12   = nn.LeakyReLU(0.2)        

        self.s_up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu21   = nn.LeakyReLU(0.2)                
        self.s_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu22   = nn.LeakyReLU(0.2)        

        self.s_up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu31   = nn.LeakyReLU(0.2)                
        self.s_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu32   = nn.LeakyReLU(0.2)        

        self.s_up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu41   = nn.LeakyReLU(0.2)                
        self.s_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu42   = nn.LeakyReLU(0.2)        

        self.s_up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu51   = nn.LeakyReLU(0.2)                
        self.s_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu52   = nn.LeakyReLU(0.2)    

        self.s_up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu61   = nn.LeakyReLU(0.2)                
        self.s_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu62   = nn.LeakyReLU(0.2)    

        # REC Dec
        self.r_up1       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu11   = nn.LeakyReLU(0.2)                
        self.r_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu12   = nn.LeakyReLU(0.2)        

        self.r_up2       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu21   = nn.LeakyReLU(0.2)                
        self.r_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu22   = nn.LeakyReLU(0.2)        

        self.r_up3       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu31   = nn.LeakyReLU(0.2)                
        self.r_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu32   = nn.LeakyReLU(0.2)        

        self.r_up4       = UpsampleBlock(scale=2, input_channels=out_channels*4, output_channels=out_channels*4)
        self.r_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu41   = nn.LeakyReLU(0.2)                
        self.r_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu42   = nn.LeakyReLU(0.2)        

        self.r_up5       = UpsampleBlock(scale=2, input_channels=out_channels*2, output_channels=out_channels*2)
        self.r_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu51   = nn.LeakyReLU(0.2)                
        self.r_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu52   = nn.LeakyReLU(0.2)    

        self.r_up6       = UpsampleBlock(scale=2, input_channels=out_channels, output_channels=out_channels)
        self.r_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu61   = nn.LeakyReLU(0.2)                
        self.r_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu62   = nn.LeakyReLU(0.2)  

        # Heads
        self.enc_out   = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.3), nn.Linear(512, 1))
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        self.rec_out   = nn.Conv2d(in_channels, 1, 1)
        
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv51(x))
        x6 = self.relu62(self.conv52(x))
        x = self.down6(x6)
        
        # Bottleneck
        x = self.brelu(self.bconv(x))
        x_bot = self.brelu(self.bconv(x))                
        
        # SEG Decoder
        x = self.s_up1(x_bot)
        x = self.s_drelu11(self.s_dconv11(torch.cat([x, x6], dim=1)))
        x = self.s_drelu12(self.s_dconv12(x))                

        x = self.s_up2(x)
        x = self.s_drelu21(self.s_dconv21(torch.cat([x, x5], dim=1)))
        x = self.s_drelu22(self.s_dconv22(x))              

        x = self.s_up3(x)
        x = self.s_drelu31(self.s_dconv31(torch.cat([x, x4], dim=1)))
        x = self.s_drelu32(self.s_dconv32(x))              

        x = self.s_up4(x)
        x = self.s_drelu41(self.s_dconv41(torch.cat([x, x3], dim=1)))
        x = self.s_drelu42(self.s_dconv42(x))              

        x = self.s_up5(x)
        x = self.s_drelu51(self.s_dconv51(torch.cat([x, x2], dim=1)))
        x = self.s_drelu52(self.s_dconv52(x))              

        x = self.s_up6(x)
        x = self.s_drelu61(self.s_dconv61(torch.cat([x, x1], dim=1)))
        seg_out = self.s_drelu62(self.s_dconv62(x))              

        # REC Decoder
        x = self.r_up1(x_bot)
        x = self.r_drelu11(self.r_dconv11(torch.cat([x, x6], dim=1)))
        x = self.r_drelu12(self.r_dconv12(x))                

        x = self.r_up2(x)
        x = self.r_drelu21(self.r_dconv21(torch.cat([x, x5], dim=1)))
        x = self.r_drelu22(self.r_dconv22(x))              

        x = self.r_up3(x)
        x = self.r_drelu31(self.r_dconv31(torch.cat([x, x4], dim=1)))
        x = self.r_drelu32(self.r_dconv32(x))              

        x = self.r_up4(x)
        x = self.r_drelu41(self.r_dconv41(torch.cat([x, x3], dim=1)))
        x = self.r_drelu42(self.r_dconv42(x))              

        x = self.r_up5(x)
        x = self.r_drelu51(self.r_dconv51(torch.cat([x, x2], dim=1)))
        x = self.r_drelu52(self.r_dconv52(x))              
        
        x = self.r_up6(x)
        x = self.r_drelu61(self.r_dconv61(torch.cat([x, x1], dim=1)))
        rec_out = self.r_drelu62(self.r_dconv62(x))       

        # Heads
        x_enc = self.enc_out(x_bot)   # enc에 FC 1 layer 더 있어야 편해질듯...?
        x_dec = self.dec_out(seg_out)
        x_rec = self.rec_out(rec_out)

        return x_enc, x_dec, x_rec

class Multi_Task_Discriminator_Skip_NEW(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv1    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu1    = nn.LeakyReLU(0.2)                
        self.bconv2    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu2    = nn.LeakyReLU(0.2)                

        # CLS Dec
        self.c_flatten   = nn.Flatten()
        self.c_fc        = nn.utils.spectral_norm(torch.nn.Linear(512, 512, True))
        self.c_relu      = nn.LeakyReLU(0.2)
        self.c_drop      = nn.Dropout(p=0.3)

        # SEG Dec
        self.s_up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu11   = nn.LeakyReLU(0.2)                
        self.s_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu12   = nn.LeakyReLU(0.2)        

        self.s_up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu21   = nn.LeakyReLU(0.2)                
        self.s_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu22   = nn.LeakyReLU(0.2)        

        self.s_up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu31   = nn.LeakyReLU(0.2)                
        self.s_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu32   = nn.LeakyReLU(0.2)        

        self.s_up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu41   = nn.LeakyReLU(0.2)                
        self.s_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu42   = nn.LeakyReLU(0.2)        

        self.s_up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu51   = nn.LeakyReLU(0.2)                
        self.s_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu52   = nn.LeakyReLU(0.2)    

        self.s_up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu61   = nn.LeakyReLU(0.2)                
        self.s_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu62   = nn.LeakyReLU(0.2)    

        # REC Dec
        self.r_up1       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu11   = nn.LeakyReLU(0.2)                
        self.r_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu12   = nn.LeakyReLU(0.2)        

        self.r_up2       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu21   = nn.LeakyReLU(0.2)                
        self.r_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu22   = nn.LeakyReLU(0.2)        

        self.r_up3       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu31   = nn.LeakyReLU(0.2)                
        self.r_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu32   = nn.LeakyReLU(0.2)        

        self.r_up4       = UpsampleBlock(scale=2, input_channels=out_channels*4, output_channels=out_channels*4)
        self.r_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu41   = nn.LeakyReLU(0.2)                
        self.r_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu42   = nn.LeakyReLU(0.2)        

        self.r_up5       = UpsampleBlock(scale=2, input_channels=out_channels*2, output_channels=out_channels*2)
        self.r_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu51   = nn.LeakyReLU(0.2)                
        self.r_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu52   = nn.LeakyReLU(0.2)    

        self.r_up6       = UpsampleBlock(scale=2, input_channels=out_channels, output_channels=out_channels)
        self.r_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu61   = nn.LeakyReLU(0.2)                
        self.r_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu62   = nn.LeakyReLU(0.2)  

        # Heads
        self.enc_out   = nn.Linear(512, 1)
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        self.rec_out   = nn.Conv2d(in_channels, 1, 1)
        
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv51(x))
        x6 = self.relu62(self.conv52(x))
        x = self.down6(x6)
        
        # Bottleneck
        x = self.brelu1(self.bconv1(x))
        x_bot = self.brelu2(self.bconv2(x))                
        
        # CLS Decoder
        x = self.c_flatten(x_bot)
        x = self.c_fc(x)
        x = self.c_relu(x)
        cls_out = self.c_drop(x)
        
        # SEG Decoder
        x = self.s_up1(x_bot)
        x = self.s_drelu11(self.s_dconv11(torch.cat([x, x6], dim=1)))
        x = self.s_drelu12(self.s_dconv12(x))                

        x = self.s_up2(x)
        x = self.s_drelu21(self.s_dconv21(torch.cat([x, x5], dim=1)))
        x = self.s_drelu22(self.s_dconv22(x))              

        x = self.s_up3(x)
        x = self.s_drelu31(self.s_dconv31(torch.cat([x, x4], dim=1)))
        x = self.s_drelu32(self.s_dconv32(x))              

        x = self.s_up4(x)
        x = self.s_drelu41(self.s_dconv41(torch.cat([x, x3], dim=1)))
        x = self.s_drelu42(self.s_dconv42(x))              

        x = self.s_up5(x)
        x = self.s_drelu51(self.s_dconv51(torch.cat([x, x2], dim=1)))
        x = self.s_drelu52(self.s_dconv52(x))              

        x = self.s_up6(x)
        x = self.s_drelu61(self.s_dconv61(torch.cat([x, x1], dim=1)))
        seg_out = self.s_drelu62(self.s_dconv62(x))              

        # REC Decoder
        x = self.r_up1(x_bot)
        x = self.r_drelu11(self.r_dconv11(torch.cat([x, x6], dim=1)))
        x = self.r_drelu12(self.r_dconv12(x))                

        x = self.r_up2(x)
        x = self.r_drelu21(self.r_dconv21(torch.cat([x, x5], dim=1)))
        x = self.r_drelu22(self.r_dconv22(x))              

        x = self.r_up3(x)
        x = self.r_drelu31(self.r_dconv31(torch.cat([x, x4], dim=1)))
        x = self.r_drelu32(self.r_dconv32(x))              

        x = self.r_up4(x)
        x = self.r_drelu41(self.r_dconv41(torch.cat([x, x3], dim=1)))
        x = self.r_drelu42(self.r_dconv42(x))              

        x = self.r_up5(x)
        x = self.r_drelu51(self.r_dconv51(torch.cat([x, x2], dim=1)))
        x = self.r_drelu52(self.r_dconv52(x))              
        
        x = self.r_up6(x)
        x = self.r_drelu61(self.r_dconv61(torch.cat([x, x1], dim=1)))
        rec_out = self.r_drelu62(self.r_dconv62(x))       

        # Heads
        x_enc = self.enc_out(cls_out)   
        x_dec = self.dec_out(seg_out)
        x_rec = self.rec_out(rec_out)

        return x_enc, x_dec, x_rec


class MTD_GAN(nn.Module):
    def __init__(self):
        super(MTD_GAN, self).__init__()
        # Generator
        self.Generator       = FFT_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        # self.Discriminator   = Multi_Task_Discriminator_Skip(in_channels=1, out_channels=64)
        self.Discriminator   = Multi_Task_Discriminator_Skip_NEW(in_channels=1, out_channels=64)
        

        # LOSS
        self.gan_metric_cls  = ls_gan
        self.gan_metric_seg  = ls_gan_with_weight_upgrade
        
        self.pixel_loss     = CharbonnierLoss()
        self.edge_loss      = EdgeLoss()

    # Both REC
    def d_loss(self, x, y):
        fake                             = self.Generator(x).detach()   
        real_enc,  real_dec,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_dec,  fake_rec   = self.Discriminator(fake)
    
        disc_loss    = self.gan_metric_cls(real_enc, 1.) + self.gan_metric_cls(fake_enc, 0.) + self.gan_metric_seg(real_dec, 1., x-y) + self.gan_metric_seg(fake_dec, 0., x-y)
        
        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake

        # Consistency
        rec_real_enc,  rec_real_dec,  _   = self.Discriminator(real_rec.clip(0, 1))
        rec_fake_enc,  rec_fake_dec,  _   = self.Discriminator(fake_rec.clip(0, 1))

        consist_loss_real_enc = F.mse_loss(real_enc, rec_real_enc) 
        consist_loss_real_dec = F.mse_loss(real_dec, rec_real_dec)
        consist_loss_fake_enc = F.mse_loss(fake_enc, rec_fake_enc) 
        consist_loss_fake_dec = F.mse_loss(fake_dec, rec_fake_dec)

        consist_loss = consist_loss_real_enc + consist_loss_real_dec + consist_loss_fake_enc + consist_loss_fake_dec
        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss + rec_loss + consist_loss
        loss_details = {'D/real_enc': self.gan_metric_cls(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric_cls(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric_seg(real_dec, 1., x-y),
                        'D/fake_dec': self.gan_metric_seg(fake_dec, 0., x-y),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake,
                        'D/consist_loss_real_enc': consist_loss_real_enc,
                        'D/consist_loss_real_dec': consist_loss_real_dec,
                        'D/consist_loss_fake_enc': consist_loss_fake_enc,
                        'D/consist_loss_fake_dec': consist_loss_fake_dec}     

        # return [disc_loss, rec_loss, consist_loss], loss_details           # for PCgrad      
        return total_loss, loss_details
    
    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec, _     = self.Discriminator(fake)
        
        adv_loss     = self.gan_metric_cls(gen_enc, 1.) + self.gan_metric_seg(gen_dec, 1., x-y)
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric_cls(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric_seg(gen_dec, 1., x-y), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        
        # return [adv_loss, pix_loss, edge_loss], loss_details               # for PCgrad
        return total_loss, loss_details
 





# Ablation_A - CLS Discriminator
class CLS_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv1    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu1    = nn.LeakyReLU(0.2)                
        self.bconv2    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu2    = nn.LeakyReLU(0.2)                

        # CLS Dec
        self.c_flatten   = nn.Flatten()
        self.c_fc        = nn.utils.spectral_norm(torch.nn.Linear(512, 512, True))
        self.c_relu      = nn.LeakyReLU(0.2)
        self.c_drop      = nn.Dropout(p=0.3)

        # Heads
        self.enc_out   = nn.Linear(512, 1)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv51(x))
        x6 = self.relu62(self.conv52(x))
        x = self.down6(x6)
        
        # Bottleneck
        x = self.brelu1(self.bconv1(x))
        x_bot = self.brelu2(self.bconv2(x))                          
        
        # CLS Decoder
        x = self.c_flatten(x_bot)
        x = self.c_fc(x)
        x = self.c_relu(x)
        cls_out = self.c_drop(x)

        # Heads
        x_enc = self.enc_out(cls_out)   

        return x_enc

class Ablation_A(nn.Module):
    def __init__(self):
        super(Ablation_A, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)
        # self.Generator       = FFT_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = CLS_Discriminator(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()   
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake        = self.Generator(x).detach()   
        real_enc    = self.Discriminator(y)
        fake_enc    = self.Discriminator(fake)
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(fake_enc, 0.)

        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss
        loss_details = {'D/real_enc': self.gan_metric(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric(fake_enc, 0.)}


        return total_loss, loss_details

    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc                 = self.Discriminator(fake)

        gen_loss     = self.gan_metric(gen_enc, 1.)

        adv_loss     = gen_loss
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric(gen_enc, 1.),  
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}

        return total_loss, loss_details

# Ablation_B - CLS and SEG Discriminator
class CLS_SEG_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv1    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu1    = nn.LeakyReLU(0.2)                
        self.bconv2    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu2    = nn.LeakyReLU(0.2)                            

        # CLS Dec
        self.c_flatten   = nn.Flatten()
        self.c_fc        = nn.utils.spectral_norm(torch.nn.Linear(512, 512, True))
        self.c_relu      = nn.LeakyReLU(0.2)
        self.c_drop      = nn.Dropout(p=0.3)

        # SEG Dec
        self.up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu11   = nn.LeakyReLU(0.2)                
        self.dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu12   = nn.LeakyReLU(0.2)        

        self.up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu21   = nn.LeakyReLU(0.2)                
        self.dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu22   = nn.LeakyReLU(0.2)        

        self.up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.drelu31   = nn.LeakyReLU(0.2)                
        self.dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.drelu32   = nn.LeakyReLU(0.2)        

        self.up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.drelu41   = nn.LeakyReLU(0.2)                
        self.dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.drelu42   = nn.LeakyReLU(0.2)        

        self.up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.drelu51   = nn.LeakyReLU(0.2)                
        self.dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.drelu52   = nn.LeakyReLU(0.2)    

        self.up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.drelu61   = nn.LeakyReLU(0.2)                
        self.dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.drelu62   = nn.LeakyReLU(0.2)    

        # Heads
        self.enc_out   = nn.Linear(512, 1)
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv51(x))
        x6 = self.relu62(self.conv52(x))
        x = self.down6(x6)
        
        # Bottleneck
        x = self.brelu1(self.bconv1(x))
        x_bot = self.brelu2(self.bconv2(x))                                

        # CLS Decoder
        x = self.c_flatten(x_bot)
        x = self.c_fc(x)
        x = self.c_relu(x)
        cls_out = self.c_drop(x)

        # SEG Decoder
        x = self.up1(x_bot)
        x = self.drelu11(self.dconv11(torch.cat([x, x6], dim=1)))
        x = self.drelu12(self.dconv12(x))                
        
        x = self.up2(x)
        x = self.drelu21(self.dconv21(torch.cat([x, x5], dim=1)))
        x = self.drelu22(self.dconv22(x))              
        
        x = self.up3(x)
        x = self.drelu31(self.dconv31(torch.cat([x, x4], dim=1)))
        x = self.drelu32(self.dconv32(x))              

        x = self.up4(x)
        x = self.drelu41(self.dconv41(torch.cat([x, x3], dim=1)))
        x = self.drelu42(self.dconv42(x))              

        x = self.up5(x)
        x = self.drelu51(self.dconv51(torch.cat([x, x2], dim=1)))
        x = self.drelu52(self.dconv52(x))              
        
        x = self.up6(x)
        x = self.drelu61(self.dconv61(torch.cat([x, x1], dim=1)))
        seg_out = self.drelu62(self.dconv62(x))              

        # Heads
        x_enc = self.enc_out(cls_out)   
        x_dec = self.dec_out(seg_out)

        return x_enc, x_dec

class Ablation_B(nn.Module):
    def __init__(self):
        super(Ablation_B, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = CLS_SEG_Discriminator(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()   
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                  = self.Generator(x).detach()   
        real_enc,  real_dec   = self.Discriminator(y)
        fake_enc,  fake_dec   = self.Discriminator(fake)
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.)

        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss
        loss_details = {'D/real_enc': self.gan_metric(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric(real_dec, 1.),
                        'D/fake_dec': self.gan_metric(fake_dec, 0.)}
        
        return total_loss, loss_details


    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec        = self.Discriminator(fake)

        gen_loss     = self.gan_metric(gen_enc, 1.) + self.gan_metric(gen_dec, 1.)

        adv_loss     = gen_loss
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric(gen_dec, 1.), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        

        return total_loss, loss_details

# Ablation_C - MTL_D_GAN (CLS, SEG, and REC Discriminator)
class Ablation_C(nn.Module):
    def __init__(self):
        super(Ablation_C, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        # self.Discriminator   = Multi_Task_Discriminator(in_channels=1, out_channels=64)
        # self.Discriminator   = Multi_Task_Discriminator_Skip(in_channels=1, out_channels=64)
        self.Discriminator   = Multi_Task_Discriminator_Skip_NEW(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                             = self.Generator(x).detach()   
        real_enc,  real_dec,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_dec,  fake_rec   = self.Discriminator(fake)
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.)

        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake

        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss + rec_loss
        loss_details = {'D/real_enc': self.gan_metric(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric(real_dec, 1.),
                        'D/fake_dec': self.gan_metric(fake_dec, 0.),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake}
        
        return total_loss, loss_details


    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec, _     = self.Discriminator(fake)

        adv_loss     = self.gan_metric(gen_enc, 1.) + self.gan_metric(gen_dec, 1.)
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric(gen_dec, 1.), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        

        return total_loss, loss_details

# Ablation_D - MTL_D_GAN + Seg weight
class Ablation_D(nn.Module):
    def __init__(self):
        super(Ablation_D, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        # self.Discriminator   = Multi_Task_Discriminator(in_channels=1, out_channels=64)
        # self.Discriminator   = Multi_Task_Discriminator_Skip(in_channels=1, out_channels=64)
        self.Discriminator   = Multi_Task_Discriminator_Skip_NEW(in_channels=1, out_channels=64)
        

        # LOSS
        self.gan_metric_cls  = ls_gan
        self.gan_metric_seg  = ls_gan_with_weight_upgrade
        
        self.pixel_loss     = CharbonnierLoss()
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                             = self.Generator(x).detach()   
        real_enc,  real_dec,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_dec,  fake_rec   = self.Discriminator(fake)

        disc_loss    = self.gan_metric_cls(real_enc, 1.) + self.gan_metric_cls(fake_enc, 0.) + self.gan_metric_seg(real_dec, 1., x-y) + self.gan_metric_seg(fake_dec, 0., x-y)

        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake
        
        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss + rec_loss
        loss_details = {'D/real_enc': self.gan_metric_cls(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric_cls(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric_seg(real_dec, 1., x-y),
                        'D/fake_dec': self.gan_metric_seg(fake_dec, 0., x-y),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake}
        
        return total_loss, loss_details


    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec, _     = self.Discriminator(fake)

        adv_loss     = self.gan_metric_cls(gen_enc, 1.) + self.gan_metric_seg(gen_dec, 1., x-y)
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric_cls(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric_seg(gen_dec, 1., x-y), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        
        return total_loss, loss_details

# Ablation_E - MTL_D_GAN + Seg weight + REC Consistency loss 
class Ablation_E(nn.Module):
    def __init__(self):
        super(Ablation_E, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        # self.Discriminator   = Multi_Task_Discriminator(in_channels=1, out_channels=64)
        # self.Discriminator   = Multi_Task_Discriminator_Skip(in_channels=1, out_channels=64)
        self.Discriminator   = Multi_Task_Discriminator_Skip_NEW(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric_cls  = ls_gan
        self.gan_metric_seg  = ls_gan_with_weight_upgrade
        
        self.pixel_loss     = CharbonnierLoss()
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                             = self.Generator(x).detach()   
        real_enc,  real_dec,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_dec,  fake_rec   = self.Discriminator(fake)
    
        disc_loss    = self.gan_metric_cls(real_enc, 1.) + self.gan_metric_cls(fake_enc, 0.) + self.gan_metric_seg(real_dec, 1., x-y) + self.gan_metric_seg(fake_dec, 0., x-y)
        
        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake
        
        # Consistency
        rec_real_enc,  rec_real_dec,  _   = self.Discriminator(real_rec.clip(0, 1))
        rec_fake_enc,  rec_fake_dec,  _   = self.Discriminator(fake_rec.clip(0, 1))

        consist_loss_real_enc = F.mse_loss(real_enc, rec_real_enc) 
        consist_loss_real_dec = F.mse_loss(real_dec, rec_real_dec)
        consist_loss_fake_enc = F.mse_loss(fake_enc, rec_fake_enc) 
        consist_loss_fake_dec = F.mse_loss(fake_dec, rec_fake_dec)

        consist_loss = consist_loss_real_enc + consist_loss_real_dec + consist_loss_fake_enc + consist_loss_fake_dec
        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss + rec_loss + consist_loss
        loss_details = {'D/real_enc': self.gan_metric_cls(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric_cls(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric_seg(real_dec, 1., x-y),
                        'D/fake_dec': self.gan_metric_seg(fake_dec, 0., x-y),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake,
                        'D/consist_loss_real_enc': consist_loss_real_enc,
                        'D/consist_loss_real_dec': consist_loss_real_dec,
                        'D/consist_loss_fake_enc': consist_loss_fake_enc,
                        'D/consist_loss_fake_dec': consist_loss_fake_dec}     

        return total_loss, loss_details
        

    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec, _     = self.Discriminator(fake)
        
        adv_loss     = self.gan_metric_cls(gen_enc, 1.) + self.gan_metric_seg(gen_dec, 1., x-y)
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric_cls(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric_seg(gen_dec, 1., x-y), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        
        return total_loss, loss_details
 
