import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.fft as fft

from arch.Ours.skip_attention import SkipAttentionBlock, SCSEModule

# PixelUnshuffle can be used only from torch == 1.8.0 version


# Basic Unet
class Basic_UNet(nn.Module):
    def __init__(self):
        super(Basic_UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
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

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        # x = self.fc(dec1_1)
        # return x
        output = self.fc(dec1_1)
        
        output = self.relu(output + x)

        return output


# Revised Unet
class DownsampleBlock(nn.Module):
    def __init__(self, scale, input_channels):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=scale),
            nn.Conv2d(input_channels*scale**2, input_channels, kernel_size=1, stride=1, padding=1//2),
            nn.PReLU()
        )

    def forward(self, input):
        return self.downsample(input)

class UpsampleBlock(nn.Module):
    def __init__(self, scale, input_channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, input_channels*scale**2, kernel_size=1, stride=1, padding=1//2),
            nn.PixelShuffle(upscale_factor=scale),
            nn.PReLU()
        )

    def forward(self, input):
        return self.upsample(input)

class Revised_UNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(Revised_UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=input_nc, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool1 = DownsampleBlock(scale=2, input_channels=64)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = DownsampleBlock(scale=2, input_channels=128)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = DownsampleBlock(scale=2, input_channels=256)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        # self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = DownsampleBlock(scale=2, input_channels=512)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        # self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool4 = UpsampleBlock(scale=2, input_channels=512)
        
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        # self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool3 = UpsampleBlock(scale=2, input_channels=256)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        # self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool2 = UpsampleBlock(scale=2, input_channels=128)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        # self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool1 = UpsampleBlock(scale=2, input_channels=64)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

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

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        output = self.fc(dec1_1)
        output = self.relu(output + x)

        return output




# Enhanced Unet
class Enhance_UNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(Enhance_UNet, self).__init__()

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=input_nc, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool1 = DownsampleBlock(scale=2, input_channels=64, output_channels=64)  

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = DownsampleBlock(scale=2, input_channels=128, output_channels=128)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = DownsampleBlock(scale=2, input_channels=256, output_channels=256)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        # self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = DownsampleBlock(scale=2, input_channels=512, output_channels=512)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        # self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool4 = UpsampleBlock(scale=2, input_channels=512, output_channels=512)
        
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        # self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool3 = UpsampleBlock(scale=2, input_channels=256, output_channels=256)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        # self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool2 = UpsampleBlock(scale=2, input_channels=128, output_channels=128)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        # self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool1 = UpsampleBlock(scale=2, input_channels=64, output_channels=64)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=1, stride=1, padding=0, bias=True)

        # Attention module
        self.skip_attention4 = SkipAttentionBlock(F_g=512, F_l=512, F_int=512)
        self.skip_attention3 = SkipAttentionBlock(F_g=256, F_l=256, F_int=256)
        self.skip_attention2 = SkipAttentionBlock(F_g=128, F_l=128, F_int=128)
        self.skip_attention1 = SkipAttentionBlock(F_g=64,  F_l=64,  F_int=64)

        self.attention4      = SCSEModule(in_channels=256)
        self.attention3      = SCSEModule(in_channels=128)
        self.attention2      = SCSEModule(in_channels=64)
        self.attention1      = SCSEModule(in_channels=64)


    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1  = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2  = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3  = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4  = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        enc4_2  = self.skip_attention4(g=unpool4, skip=enc4_2) # skip attention
        cat4    = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2  = self.dec4_2(cat4)
        dec4_1  = self.dec4_1(dec4_2)
        dec4_1  = self.attention4(dec4_1)                      # decoder attention

        unpool3 = self.unpool3(dec4_1)
        enc3_2  = self.skip_attention3(g=unpool3, skip=enc3_2) # skip attention
        cat3    = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2  = self.dec3_2(cat3)
        dec3_1  = self.dec3_1(dec3_2)
        dec3_1  = self.attention3(dec3_1)                      # decoder attention

        unpool2 = self.unpool2(dec3_1)
        enc2_2  = self.skip_attention2(g=unpool2, skip=enc2_2) # skip attention
        cat2    = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2  = self.dec2_2(cat2)
        dec2_1  = self.dec2_1(dec2_2)
        dec2_1  = self.attention2(dec2_1)                      # decoder attention

        unpool1 = self.unpool1(dec2_1)
        enc1_2  = self.skip_attention1(g=unpool1, skip=enc1_2) # skip attention
        cat1    = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2  = self.dec1_2(cat1)
        dec1_1  = self.dec1_1(dec1_2)
        dec1_1  = self.attention1(dec1_1)                     # decoder attention

        x = self.fc(dec1_1)

        # Final Activation
        x = torch.sigmoid(x)        

        return x






# TEST
# Unet for Discriminator

from torchvision import models as torchvision_model

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision_model.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(torch.nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().to(device)
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):   
        self.vgg.eval()
        with torch.no_grad():                   
            x_vgg, y_vgg = self.vgg(x.repeat(1,3,1,1)), self.vgg(y.repeat(1,3,1,1))
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i]*self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class Perceptual_L1_Loss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_L1      = torch.nn.L1Loss()
        self.loss_VGG     = VGGLoss(device='cuda')

    def forward(self, pred=None, gt=None):

        loss_n_100 = self.loss_VGG(pred, gt) + self.loss_L1(pred, gt)

        return loss_n_100

def ls_gan(inputs, targets):
    return torch.mean((inputs - targets) ** 2)

def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(chan_out, chan_out, 3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )

class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, kernel_size=1, stride=(2 if downsample else 1))
        self.net = double_conv(input_channels, filters)
        self.down = nn.Conv2d(filters, filters, kernel_size=4, padding=1, stride=2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res

class UpBlock(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Conv2d(input_channels // 2, out_channels, kernel_size=1)
        self.conv = double_conv(input_channels, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, up):
        x = self.up(x)
        p = self.conv(torch.cat((x, up), dim=1))
        sc = self.shortcut(x)
        return p + sc

class UNet(nn.Module):
    def __init__(self, repeat_num, use_tanh=False, use_sigmoid=False, skip_connection=True, use_discriminator=True, conv_dim=64, in_channels=1):
        super().__init__()
        self.use_tanh = use_tanh
        self.skip_connection = skip_connection
        self.use_discriminator = skip_connection
        self.use_sigmoid = use_sigmoid

        filters = [in_channels] + [min(conv_dim * (2 ** i), 512) for i in range(repeat_num + 1)]
        filters[-1] = filters[-2]

        channel_in_out = list(zip(filters[:-1], filters[1:]))

        self.down_blocks = nn.ModuleList()

        for i, (in_channel, out_channel) in enumerate(channel_in_out):
            self.down_blocks.append(DownBlock(in_channel, out_channel, downsample=(i != (len(channel_in_out) - 1))))

        last_channel = filters[-1]
        if use_discriminator:
            self.to_logit = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(),
                nn.Linear(last_channel, 1)
            )

        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), channel_in_out[:-1][::-1])))
        self.conv      = double_conv(last_channel, last_channel)
        self.conv_out  = nn.Conv2d(in_channels, 1, 1)
        self.__init_weights()


    def forward(self, input):
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

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                if self.use_discriminator:
                    m.weight.data.normal_(0, 0.01)
                    if hasattr(m.bias, 'data'):
                        m.bias.data.fill_(0)
                else:
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

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

class Unet_GAN(nn.Module):
    def __init__(self):
        super(Unet_GAN, self).__init__()
        self.Generator         = Revised_UNet()
        self.Discriminator     = turn_on_spectral_norm( UNet(repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False) )
        
        self.gan_metric        = ls_gan
        self.criterion         = Perceptual_L1_Loss()

    def inference(self, x):
        with torch.no_grad():
            fake = self.Generator(x)
        return fake

    def d_loss(self, x, y):
        fake  = self.Generator(x)

        fake_enc, fake_dec     = self.Discriminator(fake.detach())
        real_enc, real_dec     = self.Discriminator(y)
        source_enc, source_dec = self.Discriminator(x)
        
        d_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + \
                 self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.) + \
                 self.gan_metric(source_enc, 0.) + self.gan_metric(source_dec, 0.)

        return d_loss


    def g_loss(self, x, y):
        fake               = self.Generator(x)
        fake_enc, fake_dec = self.Discriminator(fake)

        gan_loss           = self.gan_metric(fake_enc, 1.) + self.gan_metric(fake_dec, 1.)
        pix_loss           = self.criterion(fake, y)

        g_loss = gan_loss + 50.0*pix_loss #+ 50.0*focus_loss

        return (g_loss, gan_loss, pix_loss)