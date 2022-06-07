from re import X
import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

from module.pytorch_ssim_3d import SSIM3D
from module.piq import FID
from torchvision import models
from module.piq.feature_extractors import InceptionV3

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
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


def compute_feat(x, y, pred, device='cpu'):
    if len(x.size()) == 2:
        shape_ = x.shape[-1]
        y    = x.view(1, 1, shape_, shape_ )
        y    = y.view(1, 1, shape_, shape_ )
        pred = pred.view(1, 1, shape_, shape_ )

    feature_extractor = InceptionV3()
    feature_extractor.to(device)
    feature_extractor.eval()
    N = x.shape[0]

    x_features    = feature_extractor(x.repeat(1,3,1,1))
    y_features    = feature_extractor(y.repeat(1,3,1,1))
    pred_features = feature_extractor(pred.repeat(1,3,1,1))
    
    assert len(x_features) == 1, f"feature_encoder must return list with features from one layer. Got {len(x_features)}"
    
    return x_features[0].view(N, -1), y_features[0].view(N, -1), pred_features[0].view(N, -1)

def compute_FID(x_feats, y_feats, pred_feats):
    fid_metric  = FID()
    
    assert len(x_feats.shape) == 2 and len(y_feats.shape) == 2 and len(pred_feats.shape) == 2
    
    originial_fid  = fid_metric(x_feats, y_feats)    
    pred_fid       = fid_metric(pred_feats, y_feats)    
    gt_fid         = fid_metric(y_feats, y_feats)    

    return originial_fid, pred_fid, gt_fid

def compute_Perceptual(x, y, pred, option=True, device='cpu'):
    vgg_metric  = VGGLoss(device=device)

    assert len(x.shape) == 4 and len(y.shape) == 4 and len(pred.shape) == 4
    
    if option:
        originial_percep  = vgg_metric(x, y)    
        pred_percep       = vgg_metric(pred, y)    
        gt_percep         = vgg_metric(y, y)    
        return originial_percep, pred_percep, gt_percep

    else :   
        pred_percep       = vgg_metric(pred, y)    
        return pred_percep


# Ref: https://github.com/SSinyu/WGAN-VGG/blob/d9af4a2cf6d1f4271546e0c01847bbc38d13b910/metric.py#L7

def compute_measure(x, y, pred, data_range):
    
    original_psnr = compute_PSNR(x, y, data_range)
    original_ssim = compute_SSIM(x, y, data_range)
    original_rmse = compute_RMSE(x, y)
    
    pred_psnr     = compute_PSNR(pred, y, data_range)
    pred_ssim     = compute_SSIM(pred, y, data_range)
    pred_rmse     = compute_RMSE(pred, y)

    gt_psnr       = compute_PSNR(y, y, data_range)
    gt_ssim       = compute_SSIM(y, y, data_range)
    gt_rmse       = compute_RMSE(y, y)    

    return (original_psnr, original_ssim, original_rmse), (pred_psnr, pred_ssim, pred_rmse), (gt_psnr, gt_ssim, gt_rmse)


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2) + 1e-10 # prevent to inf value.
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2) + 1e-10 # prevent to inf value.
        return 10 * np.log10((data_range ** 2) / mse_)


def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1,1,shape_ ,shape_ )
        img2 = img2.view(1,1,shape_ ,shape_ )
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def compute_measure_3D(x, y, pred, data_range):
    ssim_loss = SSIM3D(window_size = 11)

    original_psnr = compute_PSNR(x, y, data_range)
    original_ssim = ssim_loss(x, y)
    original_rmse = compute_RMSE(x, y)
    
    pred_psnr     = compute_PSNR(pred, y, data_range)
    pred_ssim     = ssim_loss(pred, y)
    pred_rmse     = compute_RMSE(pred, y)

    gt_psnr       = compute_PSNR(y, y, data_range)
    gt_ssim       = ssim_loss(y, y)
    gt_rmse       = compute_RMSE(y, y)    

    return (original_psnr, original_ssim, original_rmse), (pred_psnr, pred_ssim, pred_rmse), (gt_psnr, gt_ssim, gt_rmse)
