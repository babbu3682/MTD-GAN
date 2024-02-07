import os 
import sys
import math
import utils
import torch
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from metrics import compute_feat, compute_FID, compute_PL, compute_TML, compute_RMSE, compute_PSNR, compute_SSIM



# Setting...!
fn_denorm         = lambda x: (x * 0.5) + 0.5
fn_tonumpy        = lambda x: x.cpu().detach().numpy().transpose(0, 2, 3, 1)


### Ours

# MTD_GAN
def train_MTD_GAN_Ours(model, data_loader, optimizer_G, optimizer_D, device, epoch, print_freq, batch_size, method_D):
    model.Generator.train(True)
    model.Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        if (method_D is not None):
            # Discriminator
            optimizer_D.zero_grad()
            model.Discriminator.zero_grad() 
            d_losses, d_loss_details = model.d_loss(input_n_20, input_n_100)
            loss_D, extra_outputs_D = method_D.backward(losses=d_losses, shared_parameters=list(model.Discriminator.shared_parameters()), task_specific_parameters=list(model.Discriminator.task_specific_parameters()), last_shared_parameters=list(model.Discriminator.last_shared_parameters()))
            optimizer_D.step()
            metric_logger.update(d_loss=sum(d_losses))
            metric_logger.update(**d_loss_details)
            # Generator
            optimizer_G.zero_grad()
            model.Generator.zero_grad()
            g_loss, g_loss_details = model.g_loss(input_n_20, input_n_100)
            g_loss.backward()
            optimizer_G.step()
            metric_logger.update(g_loss=g_loss)
            metric_logger.update(**g_loss_details)
            metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])

        else :
            # Discriminator
            optimizer_D.zero_grad()
            model.Discriminator.zero_grad() 
            d_loss, d_loss_details = model.d_loss(input_n_20, input_n_100)
            d_loss.backward()              # d_loss is tensor
            optimizer_D.step()
            metric_logger.update(d_loss=d_loss)        
            metric_logger.update(**d_loss_details)
            # Generator
            optimizer_G.zero_grad()
            model.Generator.zero_grad()
            g_loss, g_loss_details = model.g_loss(input_n_20, input_n_100)
            g_loss.backward()              # g_loss is tensor
            optimizer_G.step()
            metric_logger.update(g_loss=g_loss)
            metric_logger.update(**g_loss_details)
            metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_MTD_GAN_Ours(model, loss, data_loader, device, epoch, save_dir, print_freq):
    model.Generator.eval()
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'Valid: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        pred_n_100 = model.Generator(input_n_20)     
        
        L1_loss = loss(pred_n_100, input_n_100)        
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)     # (B, H, W, C)
    input_n_100  = fn_tonumpy(input_n_100)    # (B, H, W, C)
    pred_n_100   = fn_tonumpy(pred_n_100)     # (B, H, W, C)

    # PNG Save
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_input_n_20.png', input_n_20[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100[0].squeeze(), cmap="gray")
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_MTD_GAN_Ours(model, loss, data_loader, device, save_dir):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)

    input_features    = []
    target_features   = []
    pred_features     = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        pred_n_100 = model.Generator(input_n_20)

        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(save_dir.replace('/png/', '/dcm/') +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # dicom save folder # Abdomen
        os.makedirs(save_dir                           +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # png   save folder
  
        input_pl,   gt_pl,   pred_pl    = compute_PL(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_tml,  gt_tml,  pred_tml   = compute_TML(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_rmse, gt_rmse, pred_rmse  = compute_RMSE(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_psnr, gt_psnr, pred_psnr  = compute_PSNR(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_ssim, gt_ssim, pred_ssim  = compute_SSIM(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        
        input_feat, target_feat, pred_feat = compute_feat(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_features.append(input_feat); target_features.append(target_feat); pred_features.append(pred_feat)    
        metric_logger.update(input_pl=input_pl, input_tml=input_tml, input_rmse=input_rmse, input_psnr=input_psnr, input_ssim=input_ssim)
        metric_logger.update(gt_pl=gt_pl, gt_tml=gt_tml, gt_rmse=gt_rmse, gt_psnr=gt_psnr, gt_ssim=gt_ssim)
        metric_logger.update(pred_pl=pred_pl, pred_tml=pred_tml, pred_rmse=pred_rmse, pred_psnr=pred_psnr, pred_ssim=pred_ssim)

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_gt_n_20.png'),    input_n_20.squeeze(),  cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_100'][0].split('/')[8] +'/'+batch_data['path_n_100'][0].split('/')[-1].replace('.IMA', '_gt_n_100.png'),  input_n_100.squeeze(), cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_pred_n_100.png'), pred_n_100.squeeze(),  cmap="gray")

    # FID
    input_fid, gt_fid, pred_fid = compute_FID(torch.cat(input_features, dim=0), torch.cat(target_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=input_fid, gt_fid=gt_fid, pred_fid=pred_fid)
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}



### Previous Works

# 1. CNN Based
def train_CNN_Based_Previous(model, data_loader, optimizer, device, epoch, print_freq, batch_size):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        pred_n_100 = model(input_n_20)

        loss = model.loss(pred_n_100, input_n_100)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_CNN_Based_Previous(model, loss, data_loader, device, epoch, save_dir, print_freq):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'Valid: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        pred_n_100 = model(input_n_20)

        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)     # (B, H, W, C)
    input_n_100  = fn_tonumpy(input_n_100)    # (B, H, W, C)
    pred_n_100   = fn_tonumpy(pred_n_100)     # (B, H, W, C)

    # PNG Save
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_input_n_20.png', input_n_20[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100[0].squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_CNN_Based_Previous(model, loss, data_loader, device, save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)

    input_features    = []
    target_features   = []
    pred_features     = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):

        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
                
        pred_n_100 = model(input_n_20).clip(min=0, max=1)

        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(save_dir.replace('/png/', '/dcm/') +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # dicom save folder # Abdomen
        os.makedirs(save_dir                           +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # png   save folder
  
        input_pl,   gt_pl,   pred_pl    = compute_PL(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_tml,  gt_tml,  pred_tml   = compute_TML(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_rmse, gt_rmse, pred_rmse  = compute_RMSE(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_psnr, gt_psnr, pred_psnr  = compute_PSNR(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_ssim, gt_ssim, pred_ssim  = compute_SSIM(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        
        input_feat, target_feat, pred_feat = compute_feat(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_features.append(input_feat); target_features.append(target_feat); pred_features.append(pred_feat)    
        metric_logger.update(input_pl=input_pl, input_tml=input_tml, input_rmse=input_rmse, input_psnr=input_psnr, input_ssim=input_ssim)
        metric_logger.update(gt_pl=gt_pl, gt_tml=gt_tml, gt_rmse=gt_rmse, gt_psnr=gt_psnr, gt_ssim=gt_ssim)
        metric_logger.update(pred_pl=pred_pl, pred_tml=pred_tml, pred_rmse=pred_rmse, pred_psnr=pred_psnr, pred_ssim=pred_ssim)

        # Denormalize (windowing input version)
        input_n_20   = fn_tonumpy(input_n_20)     # (B, H, W, C)
        input_n_100  = fn_tonumpy(input_n_100)    # (B, H, W, C)
        pred_n_100   = fn_tonumpy(pred_n_100)     # (B, H, W, C)

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_gt_n_20.png'),     input_n_20[0].squeeze(),  cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_100'][0].split('/')[8] +'/'+batch_data['path_n_100'][0].split('/')[-1].replace('.IMA', '_gt_n_100.png'),   input_n_100[0].squeeze(), cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_pred_n_100.png'),  pred_n_100[0].squeeze(),  cmap="gray")

    # FID
    input_fid, gt_fid, pred_fid = compute_FID(torch.cat(input_features, dim=0), torch.cat(target_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=input_fid, gt_fid=gt_fid, pred_fid=pred_fid)
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


# 2. Transformer Based  
def train_TR_Based_Previous(model, data_loader, optimizer, device, epoch, print_freq, batch_size):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
            # input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            # input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        pred_n_100 = model(input_n_20)

        if model._get_name() == "Restormer":
            loss = model.loss(pred_n_100, input_n_100)
        elif model._get_name() == "CTformer":
            loss = model.loss(pred_n_100, input_n_100)*100 + 1e-4  # to prevent 0

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_TR_Based_Previous(model, loss, data_loader, device, epoch, save_dir, print_freq):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'Valid: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=32, predictor=model, overlap=0.3, mode='constant')     

        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)     # (B, H, W, C)
    input_n_100  = fn_tonumpy(input_n_100)    # (B, H, W, C)
    pred_n_100   = fn_tonumpy(pred_n_100)     # (B, H, W, C)

    # PNG Save
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_input_n_20.png', input_n_20[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100[0].squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_TR_Based_Previous(model, loss, data_loader, device, save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)

    input_features    = []
    target_features   = []
    pred_features     = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=960, predictor=model, overlap=0.90, mode='constant')

        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(save_dir.replace('/png/', '/dcm/') +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # dicom save folder # Abdomen
        os.makedirs(save_dir                           +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # png   save folder
  
        input_pl,   gt_pl,   pred_pl    = compute_PL(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_tml,  gt_tml,  pred_tml   = compute_TML(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_rmse, gt_rmse, pred_rmse  = compute_RMSE(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_psnr, gt_psnr, pred_psnr  = compute_PSNR(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_ssim, gt_ssim, pred_ssim  = compute_SSIM(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        
        input_feat, target_feat, pred_feat = compute_feat(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_features.append(input_feat); target_features.append(target_feat); pred_features.append(pred_feat)    
        metric_logger.update(input_pl=input_pl, input_tml=input_tml, input_rmse=input_rmse, input_psnr=input_psnr, input_ssim=input_ssim)
        metric_logger.update(gt_pl=gt_pl, gt_tml=gt_tml, gt_rmse=gt_rmse, gt_psnr=gt_psnr, gt_ssim=gt_ssim)
        metric_logger.update(pred_pl=pred_pl, pred_tml=pred_tml, pred_rmse=pred_rmse, pred_psnr=pred_psnr, pred_ssim=pred_ssim)

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_100'][0].split('/')[8] +'/'+batch_data['path_n_100'][0].split('/')[-1].replace('.IMA', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

    # FID
    input_fid, gt_fid, pred_fid = compute_FID(torch.cat(input_features, dim=0), torch.cat(target_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=input_fid, gt_fid=gt_fid, pred_fid=pred_fid)
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


# 3. GAN Based
### WGAN-VGG
def train_WGAN_VGG_Previous(model, data_loader, optimizer_G, optimizer_D, device, epoch, print_freq, batch_size):
    model.Generator.train(True)
    model.Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):

        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Discriminator, 4 time more training than Generator
        optimizer_D.zero_grad()
        model.Discriminator.zero_grad()  
        for _ in range(4):
            d_loss, gp_loss = model.d_loss(input_n_20, input_n_100, gp=True, return_gp=True)
            d_loss.backward()
            optimizer_D.step()
            metric_logger.update(d_loss=d_loss, gp_loss=gp_loss)

        # Generator 
        optimizer_G.zero_grad()
        model.Generator.zero_grad()     
        g_loss, p_loss = model.g_loss(input_n_20, input_n_100, pltual=True, return_p=True)
        g_loss.backward()
        optimizer_G.step()
        metric_logger.update(g_loss=g_loss, p_loss=p_loss)

        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_WGAN_VGG_Previous(model, loss, data_loader, device, epoch, save_dir, print_freq):
    model.Generator.eval()
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'Valid: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        pred_n_100 = model.Generator(input_n_20)     
        
        L1_loss = loss(pred_n_100, input_n_100)        
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)     # (B, H, W, C)
    input_n_100  = fn_tonumpy(input_n_100)    # (B, H, W, C)
    pred_n_100   = fn_tonumpy(pred_n_100)     # (B, H, W, C)

    # PNG Save
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_input_n_20.png', input_n_20[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100[0].squeeze(), cmap="gray")
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_WGAN_VGG_Previous(model, loss, data_loader, device, save_dir):
    model.Generator.eval()
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)    

    input_features    = []
    target_features   = []
    pred_features     = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model.Generator(input_n_20)

        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(save_dir.replace('/png/', '/dcm/') +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # dicom save folder # Abdomen
        os.makedirs(save_dir                           +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # png   save folder
  
        input_pl,   gt_pl,   pred_pl    = compute_PL(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_tml,  gt_tml,  pred_tml   = compute_TML(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_rmse, gt_rmse, pred_rmse  = compute_RMSE(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_psnr, gt_psnr, pred_psnr  = compute_PSNR(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_ssim, gt_ssim, pred_ssim  = compute_SSIM(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        
        input_feat, target_feat, pred_feat = compute_feat(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_features.append(input_feat); target_features.append(target_feat); pred_features.append(pred_feat)    
        metric_logger.update(input_pl=input_pl, input_tml=input_tml, input_rmse=input_rmse, input_psnr=input_psnr, input_ssim=input_ssim)
        metric_logger.update(gt_pl=gt_pl, gt_tml=gt_tml, gt_rmse=gt_rmse, gt_psnr=gt_psnr, gt_ssim=gt_ssim)
        metric_logger.update(pred_pl=pred_pl, pred_tml=pred_tml, pred_rmse=pred_rmse, pred_psnr=pred_psnr, pred_ssim=pred_ssim)

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_100'][0].split('/')[8] +'/'+batch_data['path_n_100'][0].split('/')[-1].replace('.IMA', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

    # FID
    input_fid, gt_fid, pred_fid = compute_FID(torch.cat(input_features, dim=0), torch.cat(target_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=input_fid, gt_fid=gt_fid, pred_fid=pred_fid)
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

### MAP-NN
def train_MAP_NN_Previous(model, data_loader, optimizer_G, optimizer_D, device, epoch, print_freq, batch_size):
    model.Generator.train(True)
    model.Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):

        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        # Discriminator, 4 time more training than Generator
        optimizer_D.zero_grad()
        model.Discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(4):
            d_loss, gp_loss = model.d_loss(input_n_20, input_n_100, gp=True, return_gp=True)
            d_loss.backward()
            optimizer_D.step()
            metric_logger.update(d_loss=d_loss, gp_loss=gp_loss)

        # Generator, pltual loss
        optimizer_G.zero_grad()
        model.Generator.zero_grad()     # Same as optimizer zero grad()
        g_loss, adv_loss, mse_loss, edge_loss = model.g_loss(input_n_20, input_n_100)
        g_loss.backward()
        optimizer_G.step()
        metric_logger.update(g_loss=g_loss, adv_loss=adv_loss, mse_loss=mse_loss, edge_loss=edge_loss)
        
        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_MAP_NN_Previous(model, loss, data_loader, device, epoch, save_dir, print_freq):
    model.Generator.eval()
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'Valid: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        pred_n_100 = model.Generator(input_n_20)     
            
        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)     # (B, H, W, C)
    input_n_100  = fn_tonumpy(input_n_100)    # (B, H, W, C)
    pred_n_100   = fn_tonumpy(pred_n_100)     # (B, H, W, C)

    # PNG Save
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_input_n_20.png', input_n_20[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100[0].squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_MAP_NN_Previous(model, loss, data_loader, device, save_dir):
    model.Generator.eval()
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)    

    input_features    = []
    target_features   = []
    pred_features     = []

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model.Generator(input_n_20)

        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(save_dir.replace('/png/', '/dcm/') +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # dicom save folder # Abdomen
        os.makedirs(save_dir                           +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # png   save folder
  
        input_pl,   gt_pl,   pred_pl    = compute_PL(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_tml,  gt_tml,  pred_tml   = compute_TML(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_rmse, gt_rmse, pred_rmse  = compute_RMSE(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_psnr, gt_psnr, pred_psnr  = compute_PSNR(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_ssim, gt_ssim, pred_ssim  = compute_SSIM(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        
        input_feat, target_feat, pred_feat = compute_feat(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_features.append(input_feat); target_features.append(target_feat); pred_features.append(pred_feat)    
        metric_logger.update(input_pl=input_pl, input_tml=input_tml, input_rmse=input_rmse, input_psnr=input_psnr, input_ssim=input_ssim)
        metric_logger.update(gt_pl=gt_pl, gt_tml=gt_tml, gt_rmse=gt_rmse, gt_psnr=gt_psnr, gt_ssim=gt_ssim)
        metric_logger.update(pred_pl=pred_pl, pred_tml=pred_tml, pred_rmse=pred_rmse, pred_psnr=pred_psnr, pred_ssim=pred_ssim)

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_100'][0].split('/')[8] +'/'+batch_data['path_n_100'][0].split('/')[-1].replace('.IMA', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

    # FID
    input_fid, gt_fid, pred_fid = compute_FID(torch.cat(input_features, dim=0), torch.cat(target_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=input_fid, gt_fid=gt_fid, pred_fid=pred_fid)
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

### DUGAN
def train_DUGAN_Previous(model, data_loader, optimizer_G, optimizer_Img_D, optimizer_Grad_D, device, epoch, print_freq, batch_size):
    model.Generator.train(True)
    model.Image_Discriminator.train(True)
    model.Grad_Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):

        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        # Discriminator
            # Image D
        optimizer_Img_D.zero_grad()
        model.Image_Discriminator.zero_grad() 
        Img_d_loss, Img_detail = model.Image_d_loss(input_n_20, input_n_100)
        Img_d_loss.backward()
        optimizer_Img_D.step()
        metric_logger.update(d_img_loss=Img_d_loss.item())
        metric_logger.update(**Img_detail)
            # Grad D
        optimizer_Grad_D.zero_grad()
        model.Grad_Discriminator.zero_grad()        
        Grad_d_loss, Grad_detail = model.Grad_d_loss(input_n_20, input_n_100)
        Grad_d_loss.backward()
        optimizer_Grad_D.step()
        metric_logger.update(d_grad_loss=Grad_d_loss.item())
        metric_logger.update(**Grad_detail)

        # Generator
        optimizer_G.zero_grad()
        model.Generator.zero_grad()
        g_loss, g_detail = model.g_loss(input_n_20, input_n_100)
        g_loss.backward()        
        optimizer_G.step()
        metric_logger.update(g_loss=g_loss.item())
        metric_logger.update(**g_detail)

        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_DUGAN_Previous(model, loss, data_loader, device, epoch, save_dir, print_freq):
    model.Generator.eval()
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'Valid: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        pred_n_100 = model.Generator(input_n_20)     
            
        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)     # (B, H, W, C)
    input_n_100  = fn_tonumpy(input_n_100)    # (B, H, W, C)
    pred_n_100   = fn_tonumpy(pred_n_100)     # (B, H, W, C)

    # PNG Save
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_input_n_20.png', input_n_20[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100[0].squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_DUGAN_Previous(model, loss, data_loader, device, save_dir):
    model.Generator.eval()
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)    
    
    input_features    = []
    target_features   = []
    pred_features     = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model.Generator(input_n_20)

        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(save_dir.replace('/png/', '/dcm/') +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # dicom save folder # Abdomen
        os.makedirs(save_dir                           +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # png   save folder
  
        input_pl,   gt_pl,   pred_pl    = compute_PL(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_tml,  gt_tml,  pred_tml   = compute_TML(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_rmse, gt_rmse, pred_rmse  = compute_RMSE(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_psnr, gt_psnr, pred_psnr  = compute_PSNR(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_ssim, gt_ssim, pred_ssim  = compute_SSIM(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        
        input_feat, target_feat, pred_feat = compute_feat(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_features.append(input_feat); target_features.append(target_feat); pred_features.append(pred_feat)    
        metric_logger.update(input_pl=input_pl, input_tml=input_tml, input_rmse=input_rmse, input_psnr=input_psnr, input_ssim=input_ssim)
        metric_logger.update(gt_pl=gt_pl, gt_tml=gt_tml, gt_rmse=gt_rmse, gt_psnr=gt_psnr, gt_ssim=gt_ssim)
        metric_logger.update(pred_pl=pred_pl, pred_tml=pred_tml, pred_rmse=pred_rmse, pred_psnr=pred_psnr, pred_ssim=pred_ssim)

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_100'][0].split('/')[8] +'/'+batch_data['path_n_100'][0].split('/')[-1].replace('.IMA', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

    # FID
    input_fid, gt_fid, pred_fid = compute_FID(torch.cat(input_features, dim=0), torch.cat(target_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=input_fid, gt_fid=gt_fid, pred_fid=pred_fid)
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


# 4. DN Based
def train_DN_Previous(model, data_loader, optimizer, device, epoch, print_freq, batch_size):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)
    model.scheduler.set_timesteps(num_inference_steps=1000)
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
                
        # Generate random noise
        noise = torch.randn_like(input_n_100).to(device)

        # Create timesteps
        timesteps = torch.randint(0, model.inferer.scheduler.num_train_timesteps, (input_n_100.shape[0],), device=input_n_100.device).long()

        # Get model prediction
        noise_pred = model.inferer(inputs=input_n_100, diffusion_model=model.diffusion_unet, noise=noise, timesteps=timesteps, condition=input_n_20, mode='concat')

        loss = model.criterion(noise_pred.float(), noise.float())
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_DN_Previous(model, loss, data_loader, device, epoch, save_dir, print_freq):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'Valid: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()            
        
        # Sampling image during training
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=16, predictor=model.forward, overlap=0.5, mode='constant')
        L1_loss    = loss(pred_n_100, input_n_100)
        metric_logger.update(L1_loss=L1_loss.item())

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)     # (B, H, W, C)
    input_n_100  = fn_tonumpy(input_n_100)    # (B, H, W, C)
    pred_n_100   = fn_tonumpy(pred_n_100)     # (B, H, W, C)

    # PNG Save
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_input_n_20.png', input_n_20[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100[0].squeeze(), cmap="gray")
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100[0].squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_DN_Previous(model, loss, data_loader, device, save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)

    input_features    = []
    target_features   = []
    pred_features     = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        # Sampling image during training
        pred_n_100 = model.forward(input_n_20)

        L1_loss = loss(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(save_dir.replace('/png/', '/dcm/') +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # dicom save folder # Abdomen
        os.makedirs(save_dir                           +'/'+ batch_data['path_n_20'][0].split('/')[8], mode=0o777, exist_ok=True) # png   save folder
  
        input_pl,   gt_pl,   pred_pl    = compute_PL(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_tml,  gt_tml,  pred_tml   = compute_TML(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_rmse, gt_rmse, pred_rmse  = compute_RMSE(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_psnr, gt_psnr, pred_psnr  = compute_PSNR(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        input_ssim, gt_ssim, pred_ssim  = compute_SSIM(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1))
        
        input_feat, target_feat, pred_feat = compute_feat(input=input_n_20, target=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        input_features.append(input_feat); target_features.append(target_feat); pred_features.append(pred_feat)    
        metric_logger.update(input_pl=input_pl, input_tml=input_tml, input_rmse=input_rmse, input_psnr=input_psnr, input_ssim=input_ssim)
        metric_logger.update(gt_pl=gt_pl, gt_tml=gt_tml, gt_rmse=gt_rmse, gt_psnr=gt_psnr, gt_ssim=gt_ssim)
        metric_logger.update(pred_pl=pred_pl, pred_tml=pred_tml, pred_rmse=pred_rmse, pred_psnr=pred_psnr, pred_ssim=pred_ssim)

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_100'][0].split('/')[8] +'/'+batch_data['path_n_100'][0].split('/')[-1].replace('.IMA', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(save_dir +'/'+ batch_data['path_n_20'][0].split('/')[8]  +'/'+batch_data['path_n_20'][0].split('/')[-1].replace('.IMA', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

    # FID
    input_fid, gt_fid, pred_fid = compute_FID(torch.cat(input_features, dim=0), torch.cat(target_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=input_fid, gt_fid=gt_fid, pred_fid=pred_fid)
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


