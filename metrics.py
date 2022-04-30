import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

from module.pytorch_ssim_3d import SSIM3D


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
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
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