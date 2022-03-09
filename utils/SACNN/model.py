'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-07-08 10:24:49
@LastEditors: GuoYi
'''
import torch.nn.functional as F
from torch import nn
import torch 
import numpy as np 
import time 
from .model_function import SA_Block, Conv3D_Block, AE_Conv2D_Block
from .utils import updata_ae
from torchvision.models import vgg19


"""
Generator
"""
##******************************************************************************************************************************
class SACNN_Generator(nn.Module):
    def __init__(self):
        super(SACNN_Generator, self).__init__()

        self.lay1 = Conv3D_Block(in_ch=1, out_ch=64)

        self.lay2 = Conv3D_Block(in_ch=64, out_ch=32)
        self.att1 = SA_Block(in_ch=32)        
        
        self.lay4 = Conv3D_Block(in_ch=32, out_ch=16)
        self.att2 = SA_Block(in_ch=16)        
        
        self.lay6 = Conv3D_Block(in_ch=16, out_ch=32)
        self.att3 = SA_Block(in_ch=32)        

        self.lay8 = Conv3D_Block(in_ch=32, out_ch=64)

        self.head = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, padding=(0,1,1))

        

    def forward(self, x):
        # Conv3D input must be (B, C, D, H, W)
        # print("c=== ", x.shape) # [2, 1, 3, 64, 64]
        x = self.lay1(x)

        x = self.lay2(x)
        x = self.att1(x)

        x = self.lay4(x)
        x = self.att2(x)

        x = self.lay6(x)
        x = self.att3(x)

        x = self.lay8(x)

        x = self.head(x)

        x = x.squeeze(2)

        return F.relu(x)


"""
AE Loss
"""
##******************************************************************************************************************************
class AutoEncoder_2D(nn.Module):
    def __init__(self):
        super(AutoEncoder_2D, self).__init__()

        self.lay1     = AE_Conv2D_Block(in_channels=1, out_channels=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lay2     = AE_Conv2D_Block(in_channels=64, out_channels=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lay3     = AE_Conv2D_Block(in_channels=128, out_channels=256)
        self.lay4     = AE_Conv2D_Block(in_channels=256, out_channels=256)

        self.lay5     = AE_Conv2D_Block(in_channels=256, out_channels=256)
        self.lay6     = AE_Conv2D_Block(in_channels=256, out_channels=128)
        self.deconv1  = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.lay7     = AE_Conv2D_Block(in_channels=128, out_channels=64)
        self.deconv2  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.lay8     = AE_Conv2D_Block(in_channels=64, out_channels=1)
        
        
    def forward(self, x):
        x = self.lay1(x)
        x = self.maxpool1(x)
        x = self.lay2(x)
        x = self.maxpool2(x)
        x = self.lay3(x)
        y = self.lay4(x)

        x = self.lay5(y)
        x = self.lay6(x)
        x = self.deconv1(x)
        x = self.lay7(x)
        x = self.deconv2(x)
        x = self.lay8(x)

        return F.relu(x)

    def feat_extractor(self, x):
        self.eval()
        with torch.no_grad():
            x = self.lay1(x)
            x = self.maxpool1(x)
            x = self.lay2(x)
            x = self.maxpool2(x)
            x = self.lay3(x)
            y = self.lay4(x)

        return y


"""
Discriminator
"""
##******************************************************************************************************************************
# class DISC(nn.Module):
#     def __init__(self):
#         super(DISC, self).__init__()

#         self.lay1 = Conv3D_Block(in_ch=1, out_ch=64)
#         self.lay2 = Conv3D_Block(in_ch=64, out_ch=64)

#         self.lay3 = Conv3D_Block(in_ch=64, out_ch=128)
#         self.lay4 = Conv3D_Block(in_ch=128, out_ch=128)

#         self.lay5 = Conv3D_Block(in_ch=128, out_ch=256)
#         self.lay6 = Conv3D_Block(in_ch=256, out_ch=256)

#         self.fc1 = nn.Linear(256*3*64*64, 1024)    ## input:N*C*D*H*W = N*(256*3*64*64)
#         self.fc2 = nn.Linear(1024, 1)

#     def forward(self, x):
#         x = self.lay1(x)
#         x = self.lay2(x)
#         x = self.lay3(x)
#         x = self.lay4(x)
#         x = self.lay5(x)
#         x = self.lay6(x)

#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)

#         return x


class DISC(nn.Module):
    def __init__(self):
        super(DISC, self).__init__()

        self.lay1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lay2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.lay3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.lay4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.lay5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.lay6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.fc1  = nn.Linear(256*1*64*64, 1024)    ## input:N*C*D*H*W = N*(256*3*64*64)
        self.fc2  = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = self.lay6(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

"""
Whole Network
"""
##******************************************************************************************************************************
class SACNN(nn.Module):
    def __init__(self):
        super(SACNN, self).__init__()
        if torch.cuda.is_available():
            self.Generator     = SACNN_Generator().cuda()
            self.Discriminator = DISC().cuda()
            self.p_criterion   = nn.MSELoss().cuda()
        else:
            self.Generator     = SACNN_Generator()
            self.Discriminator = DISC()
            self.p_criterion   = nn.MSELoss()
        
        self.AutoeEncoder = AutoEncoder_2D()

        # pre-trained feat extractor
        print("Load feature extractor...!")
        checkpoint = torch.load("/workspace/sunggu/4.Dose_img2img/model/[Privious]SACNN_AutoEncoder_NEW/epoch_998_checkpoint.pth", map_location='cpu')
        self.AutoeEncoder.load_state_dict(checkpoint['model_state_dict'])
        for p in self.AutoeEncoder.parameters():
            p.requires_grad = False


    def d_loss(self, x, y, gp=True, return_gp=False):
        """
        discriminator loss
        """
        fake   = self.Generator(x)
        d_real = self.Discriminator(y)
        d_fake = self.Discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)

        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    def g_loss(self, x, y, perceptual=True, return_p=False):
        """
        generator loss
        """
        fake   = self.Generator(x)
        d_fake = self.Discriminator(fake)
        g_loss = -torch.mean(d_fake)
        mse_loss = self.p_criterion(x, y)
        g_loss += mse_loss
        
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + (0.1 * p_loss)
        else:
            p_loss = None
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    def p_loss(self, x, y):
        """
        percetual loss
        """

        fake = self.Generator(x)
        real = y

        # B, C, D, H, W = fake.shape
        # fake = fake.transpose(1, 2).reshape(B*D, C, H, W)
        # real = real.transpose(1, 2).reshape(B*D, C, H, W)

        fake_feature = self.AutoeEncoder.feat_extractor(fake)
        real_feature = self.AutoeEncoder.feat_extractor(real)

        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10):
        """
        gradient penalty
        """
        assert y.size() == fake.size()
        # a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1, 1)))
        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        if torch.cuda.is_available():
            a = a.cuda()

        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.Discriminator(interp)

        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty
