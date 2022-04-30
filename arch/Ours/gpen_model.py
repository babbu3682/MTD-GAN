'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import math
import random
import functools
import operator
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from .gpen_op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2, device='cpu'):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad, device=self.device)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2, device='cpu'):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad, device=self.device)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, device='cpu'):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad, device=self.device)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation
        self.device = device

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul, device=self.device)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        device='cpu'
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor, device=device)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), device=device)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self, isconcat=True):
        super().__init__()

        self.isconcat = isconcat
        self.weight   = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        if self.isconcat:
            return torch.cat((image, self.weight * noise), dim=1)
        else:
            # print("1 = " , image.shape)       torch.Size([48, 512, 4, 4])
            # print("2 = " ,self.weight.shape)  torch.Size([1])
            # print("3 = " ,noise.shape)        torch.Size([48, 1, 4, 4])
            return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        isconcat=True,
        device='cpu'
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            device=device
        )

        self.noise = NoiseInjection(isconcat)
        #self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        #self.activate = ScaledLeakyReLU(0.2)
        feat_multiplier = 2 if isconcat else 1
        self.activate = FusedLeakyReLU(out_channel*feat_multiplier, device=device)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToGray(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], device='cpu'):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel, device=device)

        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False, device=device)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        device='cpu'
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), device=device))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride, bias=bias and not activate,))

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel, device=device))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], device='cpu'):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, device=device)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)

        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

# Original
class GPEN_Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cpu'
    ):
        super().__init__()

        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu', device=device
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.input    = ConstantInput(self.channels[4])
        self.conv1    = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device)
        self.to_gray1 = ToGray(self.channels[4]*self.feat_multiplier, style_dim, upsample=False, device=device)

        self.log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_grays = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel*self.feat_multiplier, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel, isconcat=isconcat, device=device))
            self.convs.append(StyledConv(out_channel*self.feat_multiplier, out_channel, 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device))
            self.to_grays.append(ToGray(out_channel*self.feat_multiplier, style_dim, device=device))
            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2



    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            '''
            noise = [None] * (2 * (self.log_size - 2) + 1)
            '''
            noise = []
            batch = styles[0].shape[0]
            for i in range(self.n_mlp + 1):
                size = 2 ** (i+2)
                noise.append(torch.randn(batch, self.channels[size], size, size, device=styles[0].device))

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # Main generation
        out  = self.input(latent)  # constant input
        out  = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_gray1(out, latent[:, 1])
        # print("c3 == ", out.shape)  torch.Size([48, 1024, 4, 4])

        i = 1
        cnt = 0
        # for 문 4번
        for conv1, conv2, noise1, noise2, to_gray in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_grays):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_gray(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image

class GPEN_FullGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cpu'
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.log_size = int(math.log(size, 2))
        
        # Decoder
        self.generator = GPEN_Generator(size, style_dim, n_mlp, channel_multiplier=channel_multiplier, blur_kernel=blur_kernel, lr_mlp=lr_mlp, isconcat=isconcat, narrow=narrow, device=device)
        
        # Encoder
        conv = [ConvLayer(1, channels[size], 1, device=device)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]

        self.names = ['ecd%d'%i for i in range(self.log_size-1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)] 
            setattr(self, self.names[self.log_size-i+1], nn.Sequential(*conv))
            in_channel = out_channel

        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, style_dim, activation='fused_lrelu', device=device))

    def forward(self,
        inputs,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
    ):
        # Encoder
        noise = []
        conditions = []

        for i in range(self.log_size-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)
            # print(inputs.shape)
            # torch.Size([64, 512, 64, 64])
            # torch.Size([64, 512, 32, 32])
            # torch.Size([64, 512, 16, 16])
            # torch.Size([64, 512, 8, 8])
            # torch.Size([64, 512, 4, 4])

        inputs = inputs.view(inputs.shape[0], -1)   # 그냥 단순히 Flatten
        outs   = self.final_linear(inputs)

        # Decoder
        # print(outs.shape) torch.Size([32, 512])
        noise = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noise))[::-1]
        # print("C1 == ", len(noise))      10
        # print("C2 == ", noise[0].shape)  torch.Size([48, 512, 4, 4])
        # print("C3 == ", noise[1].shape)  torch.Size([48, 512, 4, 4])
        # print("C4 == ", noise[2].shape)  torch.Size([48, 512, 8, 8])
        # print("C5 == ", noise[-1].shape) torch.Size([48, 512, 64, 64])

        outs = self.generator([outs], return_latents, inject_index, truncation, truncation_latent, input_is_latent, noise=noise[1:])
        return outs

class GPEN_Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], narrow=1, device='cpu'):
        super().__init__()

        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        convs = [ConvLayer(1, channels[size], 1, device=device)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel, device=device))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, device=device)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu', device=device),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out


# Ours GFP model    
from .CMT import *

class Simple_Conv_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = ConvLayer(1,    32, 1, downsample=False, device='cuda')
        self.conv2 = ConvLayer(32,   64, 3, downsample=True,  device='cuda')
        self.conv3 = ConvLayer(64,  128, 3, downsample=True,  device='cuda')
        self.conv4 = ConvLayer(128, 256, 3, downsample=True,  device='cuda')
        self.conv5 = ConvLayer(256, 512, 3, downsample=True,  device='cuda')
         

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)
        x4 = self.conv5(x3)

        return [x4, x3, x2, x1, x0]

class Our_Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cpu'
    ):
        super().__init__()

        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu', device=device))

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.input    = ConstantInput(self.channels[4])
        self.conv1    = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device)
        self.to_gray1 = ToGray(self.channels[4]*self.feat_multiplier, style_dim, upsample=False, device=device)

        self.log_size = int(math.log(size, 2))

        self.convs     = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_grays  = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(StyledConv(in_channel*self.feat_multiplier, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel, isconcat=isconcat, device=device))
            self.convs.append(StyledConv(out_channel*self.feat_multiplier, out_channel, 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device))
            self.to_grays.append(ToGray(out_channel*self.feat_multiplier, style_dim, device=device))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2


    def forward(
        self,
        styles,
        conditions,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            noise = []
            for i in range(self.n_mlp + 1):
                noise.append(None)

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_gray1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_gray in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_grays):
            out = conv1(out, latent[:, i], noise=noise1)
            
            # SFT
            out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
            # print("cc2 == ", out_sft.shape)          #torch.Size([48, 256, 8, 8])
            # print("cc3 == ", conditions[i-1].shape)  #torch.Size([48, 256, 8, 8])
            out_sft = out_sft * conditions[i-1] + conditions[i]
            out = torch.cat([out_same, out_sft], dim=1)

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_gray(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image

class Ours_GPEN_FullGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=False, # noise를 안쓰기 때문에 꺼줘야한다.
        narrow=1,
        device='cpu'
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.log_size = int(math.log(size, 2))
        
        # Encoder - CMT
        # self.encoder      = CMT_S(img_size = 64, num_class = 1000)
        self.encoder      = Simple_Conv_Encoder()
        self.final_linear = nn.Sequential(EqualLinear(512*4*4, style_dim, activation='fused_lrelu', device=device))
        
        # Decoder
        self.generator = Our_Generator(size, style_dim, n_mlp, channel_multiplier=channel_multiplier, blur_kernel=blur_kernel, lr_mlp=lr_mlp, isconcat=isconcat, narrow=narrow, device=device)
        
        # for SFT modulations (scale and shift)
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        ch_list = [256, 128, 64, 32]
        for i in range(4):
            out_channels     = ch_list[i]
            sft_out_channels = 256

            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True)))

            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True)))        


    def forward(self,
        inputs,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
    ):
        conditions = []

        # Encoder
        feats_enc = self.encoder(inputs)
        z_code    = feats_enc[0]
        
        # print("1 == ", feats_enc[0].shape) #torch.Size([48, 512, 4, 4])    
        # print("2 == ", feats_enc[1].shape) #torch.Size([48, 256, 8, 8])
        # print("3 == ", feats_enc[2].shape) #torch.Size([48, 128, 16, 16])
        # print("4 == ", feats_enc[3].shape) #torch.Size([48, 64, 32, 32])
        # print("5 == ", feats_enc[4].shape) #torch.Size([48, 32, 64, 64])

        for t in range(4):
            scale = self.condition_scale[t](feats_enc[t+1])
            shift = self.condition_shift[t](feats_enc[t+1])
            # print("cc1 == ", scale.shape)
            conditions.append(scale.clone())
            conditions.append(shift.clone())

        # Make W vector style
        style_code = z_code.view(z_code.shape[0], -1)   # 그냥 단순히 Flatten
        style_code = self.final_linear(style_code)          

        # Decoder        
        outs = self.generator([style_code], conditions, return_latents, inject_index, truncation, truncation_latent, input_is_latent)
        return outs

class Ours_GPEN_Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], narrow=1, device='cpu'):
        super().__init__()

        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        convs = [ConvLayer(1, channels[size], 1, device=device)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel, device=device))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, device=device)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu', device=device),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out
