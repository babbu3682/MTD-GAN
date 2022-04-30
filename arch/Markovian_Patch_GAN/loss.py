
import torch.nn as nn
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 3.5).unsqueeze(1)
    _2D_window =_1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window.cuda()


class STD(torch.nn.Module):
    def __init__(self, window_size = 5):
        super(STD, self).__init__()
        self.window_size = window_size
        self.channel=1
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.window=create_window(self.window_size, self.channel)
        self.window.to(torch.device('cuda'))
    def forward(self, img):
        mu = F.conv2d(img, self.window, padding = self.window_size//2, groups = self.channel)
        mu_sq=mu.pow(2)
        sigma_sq = F.conv2d(img*img, self.window, padding = self.window_size//2, groups = self.channel) - mu_sq
        B,C,W,H=sigma_sq.shape
        sigma_sq=torch.flatten(sigma_sq, start_dim=1)
        noise_map = self.softmax(sigma_sq)
        noise_map=torch.reshape(noise_map,[B,C,W,H])
        return noise_map


class NCMSE(nn.Module):
    def __init__(self):
        super(NCMSE, self).__init__()
        self.std=STD()
    def forward(self, out_image, gt_image, org_image):
        loss = torch.mean(torch.mul(self.std(org_image - gt_image), torch.pow(out_image - gt_image, 2))) 
        return loss


class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return -torch.mean(pos) + torch.mean(neg)

class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)