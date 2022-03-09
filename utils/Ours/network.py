import math

import torch 
import torch.nn as nn
import torch.nn.functional as F

from .DiffAugment_pytorch import DiffAugment
from .Unet_Factory import *
from .DUGAN_wrapper import *
from .edcnn_model import *
from .Restormer_module.Restormer import *
from .Uformer.losses import CharbonnierLoss
from .Uformer.model import *

# from einops import rearrange
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_



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

class Low_UNet(UNet_DUGAN):
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

class High_UNet(UNet_DUGAN):
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
        self.noise_embed = nn.Linear(64, 256*64*64) # 여기서 늘림.

        # convmixer block depth = 8
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

# Transformer SPADE Block 
class SPADE_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(SPADE_TransformerBlock, self).__init__()

        self.norm1 = SPADE(norm_type='instance', norm_nc=dim, label_nc=1)
        self.attn  = Attention(dim, num_heads, bias)
        self.norm2 = SPADE(norm_type='instance', norm_nc=dim, label_nc=1)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, src):
        x = x + self.attn(self.norm1(x, src))
        x = x + self.ffn(self.norm2(x, src))
        return x


class SPADE_MLP(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.LayerNorm(norm_nc)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.LeakyReLU())
        self.mlp_gamma  = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta   = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        height = int(math.sqrt(x.size()[1]))
        segmap = F.interpolate(segmap, size=(height, height), mode='bicubic')
        actv   = self.mlp_shared(segmap)
        gamma  = self.mlp_gamma(actv).flatten(2).transpose(1, 2).contiguous()     # B H*W C
        beta   = self.mlp_beta(actv).flatten(2).transpose(1, 2).contiguous()      # B H*W C

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

# LeWinTransformer ############
class SPADE_LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff', se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        mlp_hidden_dim = int(dim * mlp_ratio)
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"


        self.norm1     = SPADE_MLP(norm_nc=dim, label_nc=1)
        self.attn      = WindowAttention(dim, win_size=(self.win_size, self.win_size), num_heads=num_heads,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                        token_projection=token_projection,se_layer=se_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = SPADE_MLP(norm_nc=dim, label_nc=1)
        self.mlp       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) if token_mlp=='ffn' else LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)


    def forward(self, x, src):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        shift_attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            
        shortcut = x
        x = self.norm1(x, src)

        x = x.view(B, H, W, C)
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=shift_attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x    = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H*W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x, src)))

        return x



# Restormer Transformer Block
##########################################################################
class Transformer_Generator(nn.Module):
    def __init__(self, dim=64, bias=False):
        super(Transformer_Generator, self).__init__()

        # optional Encoder
        self.img_encoder = ConvEncoder()

        # noise embedding
        self.noise_embed = nn.Linear(64, 64*64*64)
        
        # block = 6, 6, 4, 4 // head = 4, 2, 1, 1
        self.mixer_block1  = SPADE_TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block2  = SPADE_TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block3  = SPADE_TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block4  = SPADE_TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block5  = SPADE_TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block6  = SPADE_TransformerBlock(dim=dim, num_heads=4, ffn_expansion_factor=2.66, bias=bias)

        self.mixer_block7  = SPADE_TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block8  = SPADE_TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block9  = SPADE_TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block10 = SPADE_TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block11 = SPADE_TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block12 = SPADE_TransformerBlock(dim=dim, num_heads=2, ffn_expansion_factor=2.66, bias=bias)

        self.mixer_block13 = SPADE_TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block14 = SPADE_TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block15 = SPADE_TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block16 = SPADE_TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=bias)

        self.mixer_block17 = SPADE_TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block18 = SPADE_TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block19 = SPADE_TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.mixer_block20 = SPADE_TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=bias)

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
        x0 = x0.view(-1, 64, 64, 64)

        x1 = self.mixer_block1(x0, input)        # x1 == [1, 768, 128, 128]
        x2 = self.mixer_block2(x1, input)        # x2 == (B, 768*2, 128, 128)   
        x3 = self.mixer_block3(x2, input)  
        x4 = self.mixer_block4(x3, input)  
        x5 = self.mixer_block5(x4, input)  
        x6 = self.mixer_block6(x5, input)  

        x7  = self.mixer_block7(x6, input)  
        x8  = self.mixer_block8(x7, input)        # x8 == (B, 768, 128, 128)   
        x9  = self.mixer_block7(x8, input)  
        x10 = self.mixer_block8(x9, input)        # x8 == (B, 768, 128, 128)   
        x11 = self.mixer_block7(x10, input)  
        x12 = self.mixer_block8(x11, input)        # x8 == (B, 768, 128, 128)           

        x13 = self.mixer_block7(x12, input)  
        x14 = self.mixer_block8(x13, input)        # x8 == (B, 768, 128, 128)   
        x15 = self.mixer_block7(x14, input)  
        x16 = self.mixer_block8(x15, input)        # x8 == (B, 768, 128, 128)      

        x17 = self.mixer_block7(x16, input)  
        x18 = self.mixer_block8(x17, input)        # x8 == (B, 768, 128, 128)   
        x19 = self.mixer_block7(x18, input)  
        x20 = self.mixer_block8(x19, input)        # x8 == (B, 768, 128, 128)          

        output = self.head(x20)                # x8 == [1, 1, 512, 512]

        return mu, logvar, self.relu(output+input)

    def inference(self, input):        
        
        mu, logvar = self.img_encoder(input)
        z = self.reparameterize(mu, logvar)

        x0 = self.noise_embed(z)          # x0 == [1, 768, 128, 128]
        x0 = x0.view(-1, 64, 64, 64)

        x1 = self.mixer_block1(x0, input)        # x1 == [1, 768, 128, 128]
        x2 = self.mixer_block2(x1, input)        # x2 == (B, 768*2, 128, 128)   
        x3 = self.mixer_block3(x2, input)  
        x4 = self.mixer_block4(x3, input)  
        x5 = self.mixer_block5(x4, input)  
        x6 = self.mixer_block6(x5, input)  

        x7  = self.mixer_block7(x6, input)  
        x8  = self.mixer_block8(x7, input)        # x8 == (B, 768, 128, 128)   
        x9  = self.mixer_block7(x8, input)  
        x10 = self.mixer_block8(x9, input)        # x8 == (B, 768, 128, 128)   
        x11 = self.mixer_block7(x10, input)  
        x12 = self.mixer_block8(x11, input)        # x8 == (B, 768, 128, 128)           

        x13 = self.mixer_block7(x12, input)  
        x14 = self.mixer_block8(x13, input)        # x8 == (B, 768, 128, 128)   
        x15 = self.mixer_block7(x14, input)  
        x16 = self.mixer_block8(x15, input)        # x8 == (B, 768, 128, 128)      

        x17 = self.mixer_block7(x16, input)  
        x18 = self.mixer_block8(x17, input)        # x8 == (B, 768, 128, 128)   
        x19 = self.mixer_block7(x18, input)  
        x20 = self.mixer_block8(x19, input)        # x8 == (B, 768, 128, 128)          

        output = self.head(x20)                # x8 == [1, 1, 512, 512]

        return self.relu(output+input)


# Restormer - Decoder Transformer Block
##########################################################################
class Restormer_Decoder(nn.Module):
    def __init__(self, dim=48, bias=False):
        super(Restormer_Decoder, self).__init__()
        self.dim = dim
        self.noise_embed       = nn.Linear(int(dim*2**3), int(dim*2**3)*8*8) # 여기서 늘림.

        self.decoder_block4_1  = SPADE_TransformerBlock(dim=int(dim*2**3), num_heads=8, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block4_2  = SPADE_TransformerBlock(dim=int(dim*2**3), num_heads=8, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block4_3  = SPADE_TransformerBlock(dim=int(dim*2**3), num_heads=8, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block4_4  = SPADE_TransformerBlock(dim=int(dim*2**3), num_heads=8, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block4_5  = SPADE_TransformerBlock(dim=int(dim*2**3), num_heads=8, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block4_6  = SPADE_TransformerBlock(dim=int(dim*2**3), num_heads=8, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block4_7  = SPADE_TransformerBlock(dim=int(dim*2**3), num_heads=8, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block4_8  = SPADE_TransformerBlock(dim=int(dim*2**3), num_heads=8, ffn_expansion_factor=2.66, bias=bias)

        self.up4_3 = Upsample_Restormer(int(dim*2**3)) ## From Level 4 to Level 3

        self.decoder_block3_1  = SPADE_TransformerBlock(dim=int(dim*2**2), num_heads=4, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block3_2  = SPADE_TransformerBlock(dim=int(dim*2**2), num_heads=4, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block3_3  = SPADE_TransformerBlock(dim=int(dim*2**2), num_heads=4, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block3_4  = SPADE_TransformerBlock(dim=int(dim*2**2), num_heads=4, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block3_5  = SPADE_TransformerBlock(dim=int(dim*2**2), num_heads=4, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block3_6  = SPADE_TransformerBlock(dim=int(dim*2**2), num_heads=4, ffn_expansion_factor=2.66, bias=bias)

        self.up3_2 = Upsample_Restormer(int(dim*2**2)) ## From Level 3 to Level 2

        self.decoder_block2_1  = SPADE_TransformerBlock(dim=int(dim*2**1), num_heads=2, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block2_2  = SPADE_TransformerBlock(dim=int(dim*2**1), num_heads=2, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block2_3  = SPADE_TransformerBlock(dim=int(dim*2**1), num_heads=2, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block2_4  = SPADE_TransformerBlock(dim=int(dim*2**1), num_heads=2, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block2_5  = SPADE_TransformerBlock(dim=int(dim*2**1), num_heads=2, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block2_6  = SPADE_TransformerBlock(dim=int(dim*2**1), num_heads=2, ffn_expansion_factor=2.66, bias=bias)

        self.up2_1 = Upsample_Restormer(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_block1_1 = SPADE_TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block1_2 = SPADE_TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block1_3 = SPADE_TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.decoder_block1_4 = SPADE_TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66, bias=bias)

        self.refinement_1 = SPADE_TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.refinement_2 = SPADE_TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.refinement_3 = SPADE_TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66, bias=bias)
        self.refinement_4 = SPADE_TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66, bias=bias)

        self.head     = nn.Conv2d(int(dim), 1, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu     = nn.ReLU()


    def forward(self, input):
        # 다운에서 키우기
        z  = torch.randn((input.size(0), int(self.dim*2**3)), device='cuda')
        z  = self.noise_embed(z).view(-1, int(self.dim*2**3), 8, 8)       

        # D1
        x = self.decoder_block4_1(z, input) 
        x = self.decoder_block4_2(x, input)
        x = self.decoder_block4_3(x, input)
        x = self.decoder_block4_4(x, input)
        x = self.decoder_block4_5(x, input)
        x = self.decoder_block4_6(x, input)
        x = self.decoder_block4_7(x, input)
        x = self.decoder_block4_8(x, input) # torch.Size([16, 384, 64, 64])

        x = self.up4_3(x)
        # print("c1 == ", x.shape)

        # D2
        x = self.decoder_block3_1(x, input) 
        x = self.decoder_block3_2(x, input)
        x = self.decoder_block3_3(x, input)
        x = self.decoder_block3_4(x, input)
        x = self.decoder_block3_5(x, input)
        x = self.decoder_block3_6(x, input)

        x = self.up3_2(x)
        # print("c2 == ", x.shape)

        # D3
        x = self.decoder_block2_1(x, input) 
        x = self.decoder_block2_2(x, input)
        x = self.decoder_block2_3(x, input)
        x = self.decoder_block2_4(x, input)

        x = self.up2_1(x)
        # print("c3 == ", x.shape)

        # D4
        x = self.decoder_block1_1(x, input) 
        x = self.decoder_block1_2(x, input)
        x = self.decoder_block1_3(x, input)
        x = self.decoder_block1_4(x, input)
        # print("c4 == ", x.shape)
        # Refinement
        x = self.refinement_1(x, input) 
        x = self.refinement_2(x, input) 
        x = self.refinement_3(x, input) 
        x = self.refinement_4(x, input) 
        # print("c5 == ", x.shape)
        
        output = self.head(x) 

        return self.relu(output+input)

    def inference(self, input):        
        self.eval()
        with torch.no_grad():
            z  = torch.randn((input.size(0), int(self.dim*2**3)), device='cuda')
            z  = self.noise_embed(z).view(-1, int(self.dim*2**3), 8, 8)       

            # D1
            x = self.decoder_block4_1(z, input) 
            x = self.decoder_block4_2(x, input)
            x = self.decoder_block4_3(x, input)
            x = self.decoder_block4_4(x, input)
            x = self.decoder_block4_5(x, input)
            x = self.decoder_block4_6(x, input)
            x = self.decoder_block4_7(x, input)
            x = self.decoder_block4_8(x, input) # torch.Size([16, 384, 64, 64])

            x = self.up4_3(x)
            # print("c1 == ", x.shape)

            # D2
            x = self.decoder_block3_1(x, input) 
            x = self.decoder_block3_2(x, input)
            x = self.decoder_block3_3(x, input)
            x = self.decoder_block3_4(x, input)
            x = self.decoder_block3_5(x, input)
            x = self.decoder_block3_6(x, input)

            x = self.up3_2(x)
            # print("c2 == ", x.shape)

            # D3
            x = self.decoder_block2_1(x, input) 
            x = self.decoder_block2_2(x, input)
            x = self.decoder_block2_3(x, input)
            x = self.decoder_block2_4(x, input)

            x = self.up2_1(x)
            # print("c3 == ", x.shape)

            # D4
            x = self.decoder_block1_1(x, input) 
            x = self.decoder_block1_2(x, input)
            x = self.decoder_block1_3(x, input)
            x = self.decoder_block1_4(x, input)
            # print("c4 == ", x.shape)
            # Refinement
            x = self.refinement_1(x, input) 
            x = self.refinement_2(x, input) 
            x = self.refinement_3(x, input) 
            x = self.refinement_4(x, input) 
            # print("c5 == ", x.shape)
            
            output = self.head(x) 

        return self.relu(output+input)


# Uformer - Decoder Transformer Block
##########################################################################
class Uformer_Decoder(nn.Module):
    def __init__(self, img_size=64, embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 patch_norm=True, dowsample=Downsample_Uformer, upsample=Upsample_Uformer, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.img_size   = img_size
        self.embed_dim  = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio  = mlp_ratio
        self.win_size   = win_size
        self.pos_drop   = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        self.noise_embed = nn.Linear(int(embed_dim*16), int(embed_dim*16)) # 64//16 = 4

        # Bottleneck
        self.d0 = SPADE_LeWinTransformerBlock(dim=embed_dim*16, input_resolution=(img_size//16, img_size//16), num_heads=16, drop_path=conv_dpr[0]) 
        self.d1 = SPADE_LeWinTransformerBlock(dim=embed_dim*16, input_resolution=(img_size//16, img_size//16), num_heads=16, drop_path=conv_dpr[1]) 

        self.upsample_0 = upsample(embed_dim*16, embed_dim*16)

        self.d2 = SPADE_LeWinTransformerBlock(dim=embed_dim*16, input_resolution=(img_size//8, img_size//8), num_heads=16, drop_path=dec_dpr[:depths[5]][0]) 
        self.d3 = SPADE_LeWinTransformerBlock(dim=embed_dim*16, input_resolution=(img_size//8, img_size//8), num_heads=16, drop_path=dec_dpr[:depths[5]][1]) 

        
        self.upsample_1 = upsample(embed_dim*16, embed_dim*8)

        self.d4 = SPADE_LeWinTransformerBlock(dim=embed_dim*8, input_resolution=(img_size//4, img_size//4), num_heads=8, drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])][0]) 
        self.d5 = SPADE_LeWinTransformerBlock(dim=embed_dim*8, input_resolution=(img_size//4, img_size//4), num_heads=8, drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])][1]) 


        self.upsample_2 = upsample(embed_dim*8, embed_dim*4)

        self.d6 = SPADE_LeWinTransformerBlock(dim=embed_dim*4, input_resolution=(img_size//2, img_size//2), num_heads=4, drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])][0]) 
        self.d7 = SPADE_LeWinTransformerBlock(dim=embed_dim*4, input_resolution=(img_size//2, img_size//2), num_heads=4, drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])][1]) 


        self.upsample_3 = upsample(embed_dim*4, embed_dim*2)

        self.d8 = SPADE_LeWinTransformerBlock(dim=embed_dim*2, input_resolution=(img_size, img_size), num_heads=2, drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])][0]) 
        self.d9 = SPADE_LeWinTransformerBlock(dim=embed_dim*2, input_resolution=(img_size, img_size), num_heads=2, drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])][1]) 

        self.head = nn.Conv2d(embed_dim*2, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, input):

        z  = torch.randn((input.size(0), (self.img_size//16)**2, int(self.embed_dim*16)), device='cuda')
        z  = self.noise_embed(z)

        # print("c1 == ", z.shape)  # [48, 16, 512]
        
        # Bottleneck
        x0 = self.d0(z, input)
        x0 = self.d1(x0, input)
        
        up0 = self.upsample_0(x0)
        # print("c2 == ", up0.shape) # [48, 64, 512]

        x1 = self.d2(up0, input)
        x1 = self.d3(x1, input)

        up1 = self.upsample_1(x1)
        # print("c3 == ", up1.shape) # [48, 256, 256]

        x2 = self.d4(up1, input)
        x2 = self.d5(x2, input)

        up2 = self.upsample_2(x2)
        # print("c4 == ", up2.shape) # [48, 1024, 128]

        x3 = self.d6(up2, input)
        x3 = self.d7(x3, input)                        

        up3 = self.upsample_3(x3)
        # print("c5 == ", up3.shape) # [48, 4096, 64]

        x4 = self.d8(up3, input)
        x4 = self.d9(x4, input)       

        # print("c6 == ", x4.shape) # [48, 4096, 64]
        # Output Projection
        
        x4 = x4.transpose(1, 2).view(-1, int(self.embed_dim*2), self.img_size, self.img_size)
        y  = self.head(x4)

        # print("c7 == ", y.shape) # [48, 1, 64, 64]
        
        return F.relu(y)

    def inference(self, input):        
        
        z  = torch.randn((input.size(0), (self.img_size//16)**2, int(self.embed_dim*16)), device='cuda')
        z  = self.noise_embed(z)

        x0 = self.d0(z, input)
        x0 = self.d1(x0, input)
        
        up0 = self.upsample_0(x0)

        x1 = self.d2(up0, input)
        x1 = self.d3(x1, input)

        up1 = self.upsample_1(x1)

        x2 = self.d4(up1, input)
        x2 = self.d5(x2, input)

        up2 = self.upsample_2(x2)

        x3 = self.d6(up2, input)
        x3 = self.d7(x3, input)                        

        up3 = self.upsample_3(x3)

        x4 = self.d8(up3, input)
        x4 = self.d9(x4, input)       
        
        x4 = x4.transpose(1, 2).view(-1, int(self.embed_dim*2), self.img_size, self.img_size)
        y  = self.head(x4)
        
        return F.relu(y)






############################################################################################################
# CNN - Based
############################################################################################################
# 1. SPADE UNet
class Downsample_Unet(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_Unet, self).__init__()

        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelUnshuffle(2))
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)

class Upsample_Unet(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_Unet, self).__init__()

        # self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
        #                           nn.PixelShuffle(2))
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))                                  

    def forward(self, x):
        return self.body(x)

class SPADE_UNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(SPADE_UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=input_nc, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = Downsample_Unet(n_feat=64)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = Downsample_Unet(n_feat=128)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = Downsample_Unet(n_feat=256)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = Downsample_Unet(n_feat=512)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = Upsample_Unet(n_feat=512)
        
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)
        self.norm4  = SPADE_Half(norm_type='instance', norm_nc=256, label_nc=1)

        self.unpool3 = Upsample_Unet(n_feat=256)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        self.norm3  = SPADE_Half(norm_type='instance', norm_nc=128, label_nc=1)

        self.unpool2 = Upsample_Unet(n_feat=128)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
        self.norm2  = SPADE_Half(norm_type='instance', norm_nc=64, label_nc=1)

        self.unpool1 = Upsample_Unet(n_feat=64)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)
        self.norm1  = SPADE_Half(norm_type='instance', norm_nc=64, label_nc=1)

        self.fc   = nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        
        dec5_1 = self.dec5_1(enc5_1)
        
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        dec4_1 = self.norm4(dec4_1, x)
        
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        dec3_1 = self.norm3(dec3_1, x)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        dec2_1 = self.norm2(dec2_1, x)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)
        dec1_1 = self.norm1(dec1_1, x)

        output = self.fc(dec1_1)
        # output = self.relu(output + x)
        output = self.relu(output)

        return output


# 2. Upgrade
class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        B = torch.randn(in_features, out_features)*scale
        self.register_buffer("B", B)
    
    def forward(self, x):
        x_proj = torch.matmul(2*math.pi*x.permute(0,2,3,1), self.B)
        x_proj = x_proj.permute(0,3,1,2)
        out    = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1) # B, C, H, W
        return out

class SPADE_FF(nn.Module):
    def __init__(self, norm_type, norm_nc, ff_nc):
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
        self.mlp_shared = nn.Sequential(nn.Conv2d(ff_nc*2, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma  = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta   = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        # FF
        self.ff_layer = FourierLayer(in_features=1, out_features=ff_nc, scale=10)        

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bicubic')
        
        # FF
        segmap = self.ff_layer(segmap)

        actv   = self.mlp_shared(segmap)
        gamma  = self.mlp_gamma(actv)
        beta   = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADE_FF_Half(nn.Module):
    def __init__(self, norm_type, norm_nc, ff_nc):
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
        self.mlp_shared = nn.Sequential(nn.Conv2d(ff_nc*2, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma  = nn.Conv2d(nhidden, norm_nc//2, kernel_size=ks, padding=pw)
        self.mlp_beta   = nn.Conv2d(nhidden, norm_nc//2, kernel_size=ks, padding=pw)

        # FF
        self.ff_layer = FourierLayer(in_features=1, out_features=ff_nc, scale=10)        

    def forward(self, x, segmap):
        
        x_norm, x_idt = torch.chunk(x, 2, dim=1)
        
        # Part 1. generate parameter-free normalized activations
        x_norm = self.param_free_norm(x_norm)
        
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bicubic')

        # FF
        segmap = self.ff_layer(segmap)

        actv   = self.mlp_shared(segmap)
        gamma  = self.mlp_gamma(actv)
        beta   = self.mlp_beta(actv)

        # apply scale and bias
        x_norm = x_norm*gamma + beta

        out = torch.cat([x_norm, x_idt], dim=1)

        return out


class SPADE_UNet_Upgrade(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(SPADE_UNet_Upgrade, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1  = CBR2d(in_channels=input_nc, out_channels=64)
        self.norm1_1 = SPADE_FF_Half(norm_type='instance', norm_nc=64, ff_nc=64)
        self.enc1_2  = CBR2d(in_channels=64, out_channels=64)
        self.norm1_2 = SPADE_FF_Half(norm_type='instance', norm_nc=64, ff_nc=64)

        self.pool1 = Downsample_Unet(n_feat=64)

        self.enc2_1  = CBR2d(in_channels=64, out_channels=128)
        self.norm2_1 = SPADE_FF_Half(norm_type='instance', norm_nc=128, ff_nc=64)
        self.enc2_2  = CBR2d(in_channels=128, out_channels=128)
        self.norm2_2 = SPADE_FF_Half(norm_type='instance', norm_nc=128, ff_nc=64)

        self.pool2 = Downsample_Unet(n_feat=128)

        self.enc3_1  = CBR2d(in_channels=128, out_channels=256)
        self.norm3_1 = SPADE_FF_Half(norm_type='instance', norm_nc=256, ff_nc=64)
        self.enc3_2  = CBR2d(in_channels=256, out_channels=256)
        self.norm3_2 = SPADE_FF_Half(norm_type='instance', norm_nc=256, ff_nc=64)

        self.pool3 = Downsample_Unet(n_feat=256)

        self.enc4_1  = CBR2d(in_channels=256, out_channels=512)
        self.norm4_1 = SPADE_FF_Half(norm_type='instance', norm_nc=512, ff_nc=64)
        self.enc4_2  = CBR2d(in_channels=512, out_channels=512)
        self.norm4_2 = SPADE_FF_Half(norm_type='instance', norm_nc=512, ff_nc=64)

        self.pool4 = Downsample_Unet(n_feat=512)

        self.enc5_1  = CBR2d(in_channels=512, out_channels=1024)
        self.norm5_1 = SPADE_FF_Half(norm_type='instance', norm_nc=1024, ff_nc=64)
        # Expansive path
        self.dec5_1   = CBR2d(in_channels=1024, out_channels=512)
        self.dnorm5_1 = SPADE_FF(norm_type='instance', norm_nc=512, ff_nc=64)

        self.unpool4 = Upsample_Unet(n_feat=512)
        
        self.dec4_2   = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dnorm4_2 = SPADE_FF(norm_type='instance', norm_nc=512, ff_nc=64)
        self.dec4_1   = CBR2d(in_channels=512, out_channels=256)
        self.dnorm4_1 = SPADE_FF(norm_type='instance', norm_nc=256, ff_nc=64)

        self.unpool3 = Upsample_Unet(n_feat=256)

        self.dec3_2   = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dnorm3_2 = SPADE_FF(norm_type='instance', norm_nc=256, ff_nc=64)
        self.dec3_1   = CBR2d(in_channels=256, out_channels=128)
        self.dnorm3_1 = SPADE_FF(norm_type='instance', norm_nc=128, ff_nc=64)

        self.unpool2 = Upsample_Unet(n_feat=128)

        self.dec2_2   = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dnorm2_2 = SPADE_FF(norm_type='instance', norm_nc=128, ff_nc=64)
        self.dec2_1   = CBR2d(in_channels=128, out_channels=64)
        self.dnorm2_1 = SPADE_FF(norm_type='instance', norm_nc=64, ff_nc=64)

        self.unpool1 = Upsample_Unet(n_feat=64)

        self.dec1_2   = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dnorm1_2 = SPADE_FF(norm_type='instance', norm_nc=64, ff_nc=64)
        self.dec1_1   = CBR2d(in_channels=64, out_channels=64)
        self.dnorm1_1 = SPADE_FF(norm_type='instance', norm_nc=64, ff_nc=64)

        self.fc   = nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_1 = self.norm1_1(enc1_1, x)
        enc1_2 = self.enc1_2(enc1_1)
        enc1_2 = self.norm1_2(enc1_2, x)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_1 = self.norm2_1(enc2_1, x)
        enc2_2 = self.enc2_2(enc2_1)
        enc2_2 = self.norm2_2(enc2_2, x)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_1 = self.norm3_1(enc3_1, x)
        enc3_2 = self.enc3_2(enc3_1)
        enc3_2 = self.norm3_2(enc3_2, x)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_1 = self.norm4_1(enc4_1, x)
        enc4_2 = self.enc4_2(enc4_1)
        enc4_2 = self.norm4_2(enc4_2, x)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        enc5_1 = self.norm5_1(enc5_1, x)
        
        dec5_1 = self.dec5_1(enc5_1)
        dec5_1 = self.dnorm5_1(dec5_1, x)
        
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_2 = self.dnorm4_2(dec4_2, x)
        dec4_1 = self.dec4_1(dec4_2)
        dec4_1 = self.dnorm4_1(dec4_1, x)
        
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_2 = self.dnorm3_2(dec3_2, x)
        dec3_1 = self.dec3_1(dec3_2)
        dec3_1 = self.dnorm3_1(dec3_1, x)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_2 = self.dnorm2_2(dec2_2, x)
        dec2_1 = self.dec2_1(dec2_2)
        dec2_1 = self.dnorm2_1(dec2_1, x)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_2 = self.dnorm1_2(dec1_2, x)
        dec1_1 = self.dec1_1(dec1_2)
        dec1_1 = self.dnorm1_1(dec1_1, x)

        output = self.fc(dec1_1)
        # output = self.relu(output + x)
        output = self.relu(output)

        return output




############################################################################################################
# GAN - Based
############################################################################################################
# 1. FSGAN - base code
class FSGAN(nn.Module):
    def __init__(self, generator_type):
        super(FSGAN, self).__init__()
        if generator_type == 'ConvMixer':
            self.Generator      = ConvMixer_Generator(dim=256, kernel_size=9)

        elif generator_type == 'Restormer':
            self.Generator      = Transformer_Generator(dim=64, bias=False)

        elif generator_type == 'Restormer_Decoder':
            self.Generator      = Restormer_Decoder(dim=48, bias=False)            

        elif generator_type == 'Uformer_Decoder':
            self.Generator      = Uformer_Decoder()

        self.Low_discriminator  = Low_UNet(in_channels=2+1, repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False)
        self.High_discriminator = High_UNet(in_channels=20+1, repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False)
        
        # self.gan_metric         = ls_gan
        self.gan_metric         = nn.BCEWithLogitsLoss()
        self.pixel_metric       = CharbonnierLoss()
        
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

        disc_loss = self.gan_metric(low_real_enc, torch.ones_like(low_real_enc)) + self.gan_metric(low_real_dec, torch.ones_like(low_real_dec)) + \
                    self.gan_metric(low_fake_enc, torch.zeros_like(low_fake_enc)) + self.gan_metric(low_fake_dec, torch.zeros_like(low_fake_dec)) + \
                    self.gan_metric(low_source_enc, torch.zeros_like(low_source_enc)) + self.gan_metric(low_source_dec, torch.zeros_like(low_source_dec))        

        return disc_loss

    def train_High_Discriminator(self, full_dose, low_dose, gen_full_dose, prefix='High_Freq', n_iter=0):
        ############## Train Discriminator ###################
        high_real_enc,   high_real_dec    = self.normalize_gradient_enc_dec(self.High_discriminator, full_dose)
        high_fake_enc,   high_fake_dec    = self.normalize_gradient_enc_dec(self.High_discriminator, gen_full_dose.detach())
        high_source_enc, high_source_dec  = self.normalize_gradient_enc_dec(self.High_discriminator, low_dose)

        disc_loss = self.gan_metric(high_real_enc, torch.ones_like(high_real_enc)) + self.gan_metric(high_real_dec, torch.ones_like(high_real_dec)) + \
                    self.gan_metric(high_fake_enc, torch.zeros_like(high_fake_enc)) + self.gan_metric(high_fake_dec, torch.zeros_like(high_fake_dec)) + \
                    self.gan_metric(high_source_enc, torch.zeros_like(high_source_enc)) + self.gan_metric(high_source_dec, torch.zeros_like(high_source_dec))

        return disc_loss

        





