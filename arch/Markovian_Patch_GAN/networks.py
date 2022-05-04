import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


# Referemce: https://github.com/reach2sbera/ldct_nonlocal
# Warning: Even though this code is official, the STD loss is very strange. It is different with paper fomulation...

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size=window_size, sigma=3.5).unsqueeze(1)  # the sigma = 1.5 is shown in the paper value, but sigma = 3.5 is shown in the code...
    _2D_window =_1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window     = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window.cuda()

def create_mask(neighborhood_size, SIZE):
    mask=torch.zeros([SIZE,SIZE], dtype=torch.bool)
    for j in range(0, SIZE):
        for i in range(0, int(neighborhood_size*np.sqrt(SIZE)), int(np.sqrt(SIZE))):
            for k in range(neighborhood_size):
                mask[j,i+k]=1
    return mask




# Even though this code is official, the STD loss is very strange. It is different with paper fomulation...
# https://en.wikipedia.org/wiki/Standard_deviation

class STD(torch.nn.Module):
    def __init__(self, window_size=5):
        super(STD, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.window  = create_window(self.window_size, self.channel).cuda()

    def forward(self, img):
        mu         = F.conv2d(img, self.window, padding = self.window_size//2, groups = self.channel)
        mu_sq      = mu.pow(2)
        sigma_sq   = F.conv2d(img*img, self.window, padding = self.window_size//2, groups = self.channel) - mu_sq
        B, C, W, H = sigma_sq.shape
        sigma_sq   = torch.flatten(sigma_sq, start_dim=1)
        noise_map  = self.softmax(sigma_sq)
        noise_map  = torch.reshape(noise_map,[B,C,W,H])
        return noise_map


class NCMSE(nn.Module):
    def __init__(self):
        super(NCMSE, self).__init__()
        self.std=STD()

    def forward(self, out_image, gt_image, org_image):
        loss = torch.mean( torch.mul(self.std(org_image - gt_image), torch.pow(out_image - gt_image, 2)) )   # self.std(org_image - gt_image) is p(x, y) in paper.
        return loss

    
'''
# I think the below code is right.

class STD(torch.nn.Module):
    def __init__(self, window_size=5):
        super(STD, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.window  = create_window(self.window_size, self.channel).cuda()

    def forward(self, diff):
        diff_sq_mean  = F.conv2d(diff*diff, self.window, padding = self.window_size//2, groups = self.channel).mean(dim=1, keepdim=True) 
        mu_sq         = F.conv2d(diff,      self.window, padding = self.window_size//2, groups = self.channel).mean(dim=1, keepdim=True).pow(2)
        sqrt_result   = torch.sqrt(diff_sq_mean - mu_sq)
        noise_map     = self.softmax(sqrt_result)

        return noise_map


class NCMSE(nn.Module):
    def __init__(self):
        super(NCMSE, self).__init__()
        self.std=STD()

    def forward(self, out_image, gt_image, org_image):
        loss = torch.mean( torch.mul(self.std(org_image - gt_image), torch.pow(out_image - gt_image, 2)) )   # self.std(org_image - gt_image) is p(x, y) in paper.
        return loss


'''







class NonLocal(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation, SIZE):
        super(NonLocal,self).__init__()
        self.chanel_in  = in_dim
        self.activation = activation
    
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv   = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Sequential(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 3,padding=1 ),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 3,padding=1 ))            
        #self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
        self.mask     = create_mask(neighborhood_size=3, SIZE=int(SIZE))  # patch image size / (6 * 4) = 2.6666 -> "3"
        self.relu     = nn.ReLU(inplace=True)

    def forward(self,x):
        m_batchsize, C, width, height = x.size()
        proj_query    = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key      = self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy        = torch.bmm(proj_query,proj_key) # transpose check
        #
        mask          = self.mask.repeat(m_batchsize, 1, 1)
        energy[~mask ]= 0
        
        attention     = self.softmax(energy) # BX (N) X (N) 
        proj_value    = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
    
        return torch.cat((out,x),dim=1)
      
class UNet(nn.Module):
  def __init__(self, image_size=64):
    super(UNet, self).__init__()   
    self.input_channel=1
    self.inter_channel=64

    self.conv1  = nn.Sequential(nn.Conv2d(1,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.layer1 = nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.pool1  = nn.MaxPool2d(kernel_size=(2, 2))
    self.layer2 = nn.Sequential(nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.pool2  = nn.MaxPool2d(kernel_size=(2, 2))
    self.layer3 = nn.Sequential(NonLocal(64, 'relu', (image_size/4)*(image_size/4)),
		 	                 nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
       			             NonLocal(64, 'relu', image_size/4*(image_size/4)),
                             nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),                
                             NonLocal(64, 'relu', image_size/4*(image_size/4)),
                             nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True), 
                             NonLocal(64, 'relu', image_size/4*(image_size/4)),
                             nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.pool3  = nn.Upsample(scale_factor=2, mode='nearest')
    self.layer4 = nn.Sequential(nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.pool4  = nn.Upsample(scale_factor=2, mode='nearest')      
    self.layer5 = nn.Sequential(nn.Conv2d(2*self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(self.inter_channel,self.inter_channel,5,padding=2),
                             nn.ReLU(inplace=True))
    self.conv2  = nn.Conv2d(self.inter_channel,1,3,padding=1)

  def forward(self,x):

      x=self.conv1(x)

      x1=self.layer1(x)
    
      x=self.pool1(x1)
    
      x2=self.layer2(x)
    
      x=self.pool2(x2)
    
      x=self.layer3(x)
    
      x=self.pool3(x)
    

      x=self.layer4(torch.cat((x2 , x),1))

      x=self.pool4(x)

      x=self.layer5(torch.cat((x1 , x),1))

      x=self.conv2(x)

      return F.relu(x)

class SNConvWithActivation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class Self_Attn(nn.Module):
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv   = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1) #

    def forward(self,x):
        m_batchsize, C, width, height = x.size()

        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key    = self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy      = torch.bmm(proj_query,proj_key) # transpose check
        attention   = self.softmax(energy) # BX (N) X (N) 
        proj_value  = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x

        return out
           
class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(1, cnum, 3, 1),
            SNConvWithActivation(cnum, 2*cnum, 3, 2),
            SNConvWithActivation(2*cnum, 4*cnum, 3, 1),
            SNConvWithActivation(4*cnum, 8*cnum, 3, 1),
            SNConvWithActivation(8*cnum, 8*cnum, 3, 2, padding=2),
            SNConvWithActivation(8*cnum, 8*cnum, 3, padding=2),
            SNConvWithActivation(8*cnum, 8*cnum, 3, padding =2),
            Self_Attn(8*cnum),
            SNConvWithActivation(8*cnum, 8*cnum, 3, padding=2),
            Self_Attn(8*cnum),
            SNConvWithActivation(8*cnum, 8*cnum, 3, padding=2)
        )
    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        return x




class Markovian_Patch_GAN(nn.Module):
    def __init__(self):
        super(Markovian_Patch_GAN, self).__init__()
        self.Generator         = UNet(image_size=64)
        self.Discriminator     = ImageDiscriminator()
        
        # Loss
        # self.p_criterion       = NCMSE()
        

    def d_loss(self, x, y):
        fake   = self.Generator(x).detach()
        d_real = self.Discriminator(y)
        d_fake = self.Discriminator(fake)

        # d_loss = self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        return d_loss

    def g_loss(self, x, y):
        fake   = self.Generator(x)
        d_fake = self.Discriminator(fake)

        # g_loss = -torch.mean(d_fake) + 0.1*self.p_criterion(fake, y, x)
        g_loss = -torch.mean(d_fake)
        return g_loss



