import os
import numpy as np

import torch
import torch.nn as nn
# import torch.fft as fft
import sys
sys.path.append(os.path.abspath('/workspace/sunggu/4.Dose_img2img/module'))
from module.skip_attention import SkipAttentionBlock, SCSEModule
import torch.nn.functional as F

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


