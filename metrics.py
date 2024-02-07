from re import X
import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

from module.piq import FID
from torchvision import models
from module.piq.feature_extractors import InceptionV3




# Feature-based Metrics
## FID
def compute_feat(input, target, pred, device='cpu'):
    # Ref: https://github.com/photosynthesis-team/piq/blob/9948a52fc09ac5f7fb3618ce64b7086f5c3109da/piq/base.py#L18
    assert input.shape == target.shape and target.shape == pred.shape

    feature_extractor = InceptionV3()
    feature_extractor.to(device)
    feature_extractor.eval()

    input_features  = feature_extractor(input.repeat(1,3,1,1))
    target_features = feature_extractor(target.repeat(1,3,1,1))
    pred_features   = feature_extractor(pred.repeat(1,3,1,1))
    
    assert len(input_features) == 1, f"feature_encoder must return list with features from one layer. Got {len(input_features)}"
    
    return input_features[0].flatten(1), target_features[0].flatten(1), pred_features[0].flatten(1)

def compute_FID(input_feats, target_feats, pred_feats):
    fid_metric  = FID()
    assert len(input_feats.shape) == 2 and len(target_feats.shape) == 2 and len(pred_feats.shape) == 2
    
    input_fid = fid_metric(input_feats, target_feats)
    gt_fid    = fid_metric(target_feats, target_feats)
    pred_fid  = fid_metric(pred_feats, target_feats)

    return input_fid, gt_fid, pred_fid

## PL
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
    # Ref: https://github.com/geonm/EnhanceNet-Tensorflow/blob/d0e527418f8b3fd167a61c8777483259d04fc4ab/losses.py, https://github.com/NVIDIA/pix2pixHD/blob/5a2c87201c5957e2bf51d79b8acddb9cc1920b26/models/networks.py
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

def compute_PL(input, target, pred, option=True, device='cpu'):
    vgg_metric  = VGGLoss(device=device)

    assert len(input.shape) == 4 and len(target.shape) == 4 and len(pred.shape) == 4

    if option:
        input_pl  = vgg_metric(input, target)
        gt_pl     = vgg_metric(target, target)
        pred_pl   = vgg_metric(pred, target)
        return input_pl, gt_pl, pred_pl

    else :   
        pred_pl   = vgg_metric(pred, target)    
        return pred_pl

## TML
class TextureMatchingLoss(torch.nn.Module):
    # Ref: https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    def __init__(self, patch_size=16, use_patch=True, device='cuda'):
        super(TextureMatchingLoss, self).__init__()
        self.patch_size = patch_size
        self.use_patch  = use_patch
        self.unfold     = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.vgg        = Vgg19().to(device)
        self.criterion  = torch.nn.L1Loss()
        self.weights    = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def gram_matrix(self, features):
        N, C, H, W    = features.size()
        feat_reshaped = features.view(N, C, H*W)
        gram          = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
        # gram          = gram / (C*H*W) # normalize
        return gram

    def patch_resize(self, x):
        b, c, _, _ = x.shape
        # b, c*k*k, H/k*W/k
        x = self.unfold(x)
        _, _, l = x.shape
        # b, c, k, k, H/k*W/k
        x = x.view(b, c, self.patch_size, self.patch_size, -1)
        # b, H/k*W/k, c, k, k
        x = x.permute(0, 4, 1, 2, 3)
        try:
            x = x.view(-1, c, self.patch_size, self.patch_size)
        except:
            x = x.contiguous().view(-1, c, self.patch_size, self.patch_size)
        return x

    def forward(self, x, y):
        self.vgg.eval()
        with torch.no_grad():     
            x_vgg, y_vgg = self.vgg(x.repeat(1,3,1,1)), self.vgg(y.repeat(1,3,1,1))

        loss = 0    
        for i in range(len(x_vgg)):
            if self.use_patch:
                loss += self.weights[i]*self.criterion(self.gram_matrix(self.patch_resize(x_vgg[i])), self.gram_matrix(self.patch_resize(y_vgg[i].detach())))           
            else:
                loss += self.weights[i]*self.criterion(self.gram_matrix(x_vgg[i]), self.gram_matrix(y_vgg[i].detach()))

        return loss

def compute_TML(input, target, pred, option=True, device='cpu'):
    tml_metric  = TextureMatchingLoss(device=device)

    assert len(input.shape) == 4 and len(target.shape) == 4 and len(pred.shape) == 4

    if option:
        input_tml  = tml_metric(input, target)
        gt_tml     = tml_metric(target, target)
        pred_tml   = tml_metric(pred, target)
        return input_tml, gt_tml, pred_tml
    else :   
        pred_tml   = tml_metric(pred, target)    
        return pred_tml



# Pixel-based Metrics (Ref: https://github.com/SSinyu/WGAN-VGG/blob/d9af4a2cf6d1f4271546e0c01847bbc38d13b910/metric.py#L7)
## RMSE
def compute_RMSE(input, target, pred):
    mse_metric = torch.nn.MSELoss()
    assert len(input.shape) == 4 and len(target.shape) == 4 and len(pred.shape) == 4

    input_rmse  = torch.sqrt(mse_metric(input, target)).item()
    gt_rmse     = torch.sqrt(mse_metric(target, target)).item()
    pred_rmse   = torch.sqrt(mse_metric(pred, target)).item()
    return input_rmse, gt_rmse, pred_rmse

## PSNR
def compute_PSNR(input, target, pred, data_range=1.0):
    mse_metric = torch.nn.MSELoss()
    assert len(input.shape) == 4 and len(target.shape) == 4 and len(pred.shape) == 4
    
    input_mse  = mse_metric(input, target) + 1e-10 # prevent to inf value.
    input_psnr = 10 * torch.log10((data_range ** 2) / input_mse).item()

    gt_mse  = mse_metric(target, target) + 1e-10 # prevent to inf value.
    gt_psnr = 10 * torch.log10((data_range ** 2) / gt_mse).item()

    pred_mse  = mse_metric(pred, target) + 1e-10 # prevent to inf value.
    pred_psnr = 10 * torch.log10((data_range ** 2) / pred_mse).item()        

    return input_psnr, gt_psnr, pred_psnr

## SSIM
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, data_range=1.0, window_size=11, channel=1, size_average=True):
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

def compute_SSIM(input, target, pred, data_range=1.0):
    assert len(input.shape) == 4 and len(target.shape) == 4 and len(pred.shape) == 4
    
    input_ssim  = ssim(input, target, data_range)
    gt_ssim     = ssim(target, target, data_range)
    pred_ssim   = ssim(pred, target, data_range)

    return input_ssim, gt_ssim, pred_ssim



