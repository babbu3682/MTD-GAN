from fcntl import DN_DELETE
import math
import utils
import torch
from torch.nn import functional as F

import pandas as pd
import numpy as np
from pydicom import dcmread
from tqdm import tqdm
import sys
import os 
import matplotlib.pyplot as plt
from metrics import compute_measure, compute_measure_3D, compute_FID, compute_Perceptual, compute_TML, compute_feat
from monai.inferers import sliding_window_inference
from module.sliding_window_inference_multi_output import sliding_window_inference_multi_output


def dicom_denormalize(image, MIN_HU=-1024.0, MAX_HU=3072.0):
    # image = (image - 0.5) / 0.5           # Range -1.0 ~ 1.0   @ We do not use -1~1 range becuase there is no Tanh act.
    image = (MAX_HU - MIN_HU)*image + MIN_HU
    return image

def save_dicom(original_dcm_path, pred_output, save_path):
    # pydicom 으로 저장시 자동으로 -1024를 가하는 부분이 있기에 setting 해줘야 함.
    # pred_img's Range: -1024 ~ 3072
    pred_img = pred_output.copy()
    # print("before == ", pred_img.max(), pred_img.min(), pred_img.dtype) # before ==  2557.0 / -1024.0 / float32
    
    dcm = dcmread(original_dcm_path)    

    intercept = dcm.RescaleIntercept
    slope     = dcm.RescaleSlope
    
    # pred_img -= np.int16(intercept)
    pred_img -= np.float32(intercept)
    pred_img = pred_img.astype(np.int16)

    if slope != 1:
        pred_img = pred_img.astype(np.float32) / slope
        pred_img = pred_img.astype(np.int16)

    dcm.PixelData = pred_img.squeeze().tobytes()
    # dcm.PixelData = pred_img.astype('uint16').squeeze().tobytes()
    dcm.save_as(save_path)
    
    # print("after == ", pred_img.max(), pred_img.min(), pred_img.dtype)  # after ==  3581 / 0 / int16
    # print(save_path)


# Setting...!
fn_denorm         = lambda x: (x * 0.5) + 0.5
fn_tonumpy        = lambda x: x.cpu().detach().numpy().transpose(0, 2, 3, 1)
fn_tonumpy3d      = lambda x: x.cpu().detach().numpy().transpose(0, 1, 3, 4, 2)
# fn_denorm_window  = visual_windowing_V2

###################################################################             Ours                                ###################################################################
# CNN Based  ################################################
# 1.
def train_CNN_Based_Ours(model, data_loader, optimizer, device, epoch, patch_training, print_freq, batch_size):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()
        
        pred_n_100 = model(input_n_20)

        loss = model.criterion(pred_n_100, input_n_100)
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
def valid_CNN_Based_Ours(model, criterion, data_loader, device, epoch, png_save_dir, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid: [epoch:{}]'.format(epoch)
    os.makedirs(png_save_dir, mode=0o777, exist_ok=True) 

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        

        pred_n_100 = model(input_n_20)

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)
 
    # Denormalize (No windowing input version)
    # input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    # input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    # pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 
    # # PNG Save
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)
    input_n_100  = fn_tonumpy(input_n_100)
    pred_n_100   = fn_tonumpy(pred_n_100)
    # PNG Save
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_CNN_Based_Ours(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.eval()
    
    # compute PSNR, SSIM, RMSE
    ori_psnr_avg,  ori_ssim_avg,  ori_rmse_avg  = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    gt_psnr_avg,   gt_ssim_avg,   gt_rmse_avg   = 0, 0, 0

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        # pred_n_100 = model(input_n_20)
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model, overlap=0.5, mode='constant')

        os.makedirs(save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder
        
        input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
        
        # DCM Save
        save_dicom(batch_data['path_n_20'][0],  input_n_20,  save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.dcm'))        
        save_dicom(batch_data['path_n_100'][0], input_n_100, save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.dcm'))
        save_dicom(batch_data['path_n_20'][0],  pred_n_100,  save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.dcm'))        
        
        # Metric
        original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=4095.0)
        ori_psnr_avg  += original_result[0]
        ori_ssim_avg  += original_result[1]
        ori_rmse_avg  += original_result[2]
        pred_psnr_avg += pred_result[0]
        pred_ssim_avg += pred_result[1]
        pred_rmse_avg += pred_result[2]
        gt_psnr_avg   += gt_result[0]
        gt_ssim_avg   += gt_result[1]
        gt_rmse_avg   += gt_result[2]


        # PNG Save clip for windowing visualize
        input_n_20    = input_n_20.clip(min=0, max=80)
        input_n_100   = input_n_100.clip(min=0, max=80)
        pred_n_100    = pred_n_100.clip(min=0, max=80)
        plt.imsave(save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray", vmin=0, vmax=80)
        plt.imsave(save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
        plt.imsave(save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray", vmin=0, vmax=80)

    print('\n')
    print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(data_loader), ori_ssim_avg/len(data_loader), ori_rmse_avg/len(data_loader)))
    print('\n')
    print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(data_loader), pred_ssim_avg/len(data_loader), pred_rmse_avg/len(data_loader)))        
    print('\n')
    print('GT === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(gt_psnr_avg/len(data_loader), gt_ssim_avg/len(data_loader), gt_rmse_avg/len(data_loader)))        

# 2.
def progressive(epoch):
    progress_list = ['n_80', 'n_60', 'n_40', 'n_20']
    
    if epoch < 200:
        p_index = 0
    elif epoch < 400:
        p_index = 1
    elif epoch < 600:
        p_index = 2
    elif epoch < 800:
        p_index = 3
    elif epoch < 1000:
        p_index = 3
    else :
        p_index = 3

    return progress_list[p_index]

def train_CNN_Based_Ours_Progress(model, criterion, data_loader, optimizer, device, epoch, patch_training):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)
    print_freq = 10  

    p_name = progressive(epoch)
    print("Progress noise == > ", p_name)    

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):

        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i][p_name]  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data[p_name].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()
        
        pred_n_100 = model(input_n_20)
       
        loss = criterion(pred_n_100=pred_n_100, gt_100=input_n_100)

        loss_detail = None
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_CNN_Based_Ours_Progress(model, criterion, data_loader, device, epoch, save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    p_name = progressive(epoch)
    print("Progress noise == > ", p_name)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data[p_name].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # print(input_n_20.shape) # (1, 1, 512, 512)
        pred_n_100 = model(input_n_20)

        loss = criterion(pred_n_100=pred_n_100, gt_100=input_n_100)
        loss_detail = None
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
 
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    # PNG Save
    input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 

    print(save_dir+'epoch_'+str(epoch)+'_input_n_20.png')    
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_gt_n_100.png', input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_CNN_Based_Ours_Progress(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.eval()
    
    # compute PSNR, SSIM, RMSE
    ori_psnr_avg,  ori_ssim_avg,  ori_rmse_avg  = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model(input_n_20)

        # print("c1 = ", save_dir.replace('/png/', '/dcm/'))
        # print("c2 = ", batch_data['path_n_20'])
        # print("c3 = ", batch_data['path_n_20'][0])
        # print("c4 = ", batch_data['path_n_20'][0].split('/')[7])
        # print("c5 = ", batch_data['path_n_20'][0].split('/')[-1])

        os.makedirs(save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder

        
        input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
        
        # DCM Save
        save_dicom(batch_data['path_n_20'][0],  input_n_20,  save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/gt_n_20_'     + batch_data['path_n_20'][0].split('/')[-1])        
        save_dicom(batch_data['path_n_100'][0], input_n_100, save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/gt_n_100_'    + batch_data['path_n_100'][0].split('/')[-1])
        save_dicom(batch_data['path_n_20'][0],  pred_n_100,  save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]   + '/pred_n_100_'  + batch_data['path_n_20'][0].split('/')[-1])        
        
        # Metric
        original_result, pred_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=4095.0)
        ori_psnr_avg  += original_result[0]
        ori_ssim_avg  += original_result[1]
        ori_rmse_avg  += original_result[2]
        pred_psnr_avg += pred_result[0]
        pred_ssim_avg += pred_result[1]
        pred_rmse_avg += pred_result[2]

        # PNG Save clip for windowing visualize
        input_n_20    = input_n_20.clip(min=0, max=80)
        input_n_100   = input_n_100.clip(min=0, max=80)
        pred_n_100    = pred_n_100.clip(min=0, max=80)
        plt.imsave(save_dir+batch_data['path_n_20'][0].split('/')[7] +'/gt_n_20_'   +batch_data['path_n_20'][0].split('/')[-1].replace('.dcm', '.png'),  input_n_20.squeeze(),  cmap="gray", vmin=0, vmax=80)
        plt.imsave(save_dir+batch_data['path_n_100'][0].split('/')[7]+'/gt_n_100_'  +batch_data['path_n_100'][0].split('/')[-1].replace('.dcm', '.png'), input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
        plt.imsave(save_dir+batch_data['path_n_20'][0].split('/')[7] +'/pred_n_100_'+batch_data['path_n_20'][0].split('/')[-1].replace('.dcm', '.png'),  pred_n_100.squeeze(),  cmap="gray", vmin=0, vmax=80)

    print('\n')
    print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(data_loader), ori_ssim_avg/len(data_loader), ori_rmse_avg/len(data_loader)))
    print('\n')
    print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(data_loader), pred_ssim_avg/len(data_loader), pred_rmse_avg/len(data_loader)))        



# GAN Based  ################################################
# 1. FDGAN
def train_FDGAN_Ours(model, data_loader, optimizer_G, optimizer_Image_D, optimizer_Fourier_D, device, epoch, patch_training, print_freq, batch_size):
    model.Generator.train(True)
    model.Image_Discriminator.train(True)
    model.Fourier_Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()


        # Discriminator
            # Image D
        optimizer_Image_D.zero_grad()
        model.Image_Discriminator.zero_grad() 
        Img_d_loss = model.Image_d_loss(input_n_20, input_n_100)
        Img_d_loss.backward()
        optimizer_Image_D.step()
        metric_logger.update(Img_d_loss=Img_d_loss)
            # Fourier D
        optimizer_Fourier_D.zero_grad()
        model.Fourier_Discriminator.zero_grad()        
        Fourier_d_loss = model.Fourier_d_loss(input_n_20, input_n_100)
        Fourier_d_loss.backward()
        optimizer_Fourier_D.step()        
        metric_logger.update(Fourier_d_loss=Fourier_d_loss)

        # Generator
        optimizer_G.zero_grad()
        model.Generator.zero_grad()
        g_loss = model.g_loss(input_n_20, input_n_100)
        g_loss.backward()        
        optimizer_G.step()
        metric_logger.update(g_loss=g_loss)

        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_FDGAN_Ours(model, criterion, data_loader, device, epoch, png_save_dir, print_freq, batch_size):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid: [epoch:{}]'.format(epoch)
    os.makedirs(png_save_dir, mode=0o777, exist_ok=True)  

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        # pred_n_100 = model.Generator(input_n_20)     
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.5, mode='constant')
            
        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)


    # # Denormalize (No windowing input version)
    # input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    # input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    # pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 
    # # PNG Save
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)
    input_n_100  = fn_tonumpy(input_n_100)
    pred_n_100   = fn_tonumpy(pred_n_100)
    # PNG Save
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_FDGAN_Ours(model, criterion, data_loader, device, png_save_dir):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        # pred_n_100 = model.Generator(input_n_20)     
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.5, mode='constant').clip(min=0, max=1)


        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder

        # Denormalize (No windowing input version)
        input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
                
        # Metric
        original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=4095.0)
        metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # DCM Save
        save_dicom(batch_data['path_n_20'][0],  input_n_20,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.dcm'))        
        save_dicom(batch_data['path_n_100'][0], input_n_100, png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.dcm'))
        save_dicom(batch_data['path_n_20'][0],  pred_n_100,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.dcm'))        

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)
        plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.clip(min=0, max=80).squeeze(), cmap="gray", vmin=0, vmax=80)
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)

        # # Denormalize (windowing input version)
        # input_n_20    = fn_tonumpy(input_n_20)
        # input_n_100   = fn_tonumpy(input_n_100)
        # pred_n_100    = fn_tonumpy(pred_n_100)  

        # # Metric
        # original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=1.0)        
        # metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        # metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        # metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # # PNG Save clip for windowing visualize, brain:[0, 80] HU
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        # plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

        # # INFERENCE
        # os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[-1], mode=0o777, exist_ok=True) # dicom save folder
        # os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[-1], mode=0o777, exist_ok=True) # png   save folder

        # # Denormalize (windowing input version)
        # input_n_20    = fn_tonumpy(input_n_20)
        # input_n_100   = fn_tonumpy(input_n_100)
        # pred_n_100    = fn_tonumpy(pred_n_100)  

        # # Metric
        # original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=1.0)        
        # metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        # metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        # metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # # PNG Save clip for windowing visualize, brain:[0, 80] HU
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        # plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

# 2. MTD_GAN
def train_MTD_GAN_Ours(model, data_loader, optimizer_G, optimizer_D, device, epoch, patch_training, print_freq, batch_size, pcgrad):
    model.Generator.train(True)
    model.Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()

        if pcgrad:
            # Discriminator
            optimizer_D.zero_grad()
            model.Discriminator.zero_grad() 
            d_loss, d_loss_details = model.d_loss(input_n_20, input_n_100)
            optimizer_D.pc_backward(d_loss)  # d_loss is list      
            optimizer_D.step()
            metric_logger.update(d_loss=sum(d_loss))
            metric_logger.update(**d_loss_details)

            # Generator
            optimizer_G.zero_grad()
            model.Generator.zero_grad()
            g_loss, g_loss_details = model.g_loss(input_n_20, input_n_100)
            optimizer_G.pc_backward(g_loss)  # g_loss is list    
            optimizer_G.step()
            metric_logger.update(g_loss=sum(g_loss))
            metric_logger.update(**g_loss_details)

            metric_logger.update(lr=optimizer_G.optimizer.param_groups[0]["lr"])
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
def valid_MTD_GAN_Ours(model, criterion, data_loader, device, epoch, png_save_dir, print_freq, batch_size):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid: [epoch:{}]'.format(epoch)
    os.makedirs(png_save_dir, mode=0o777, exist_ok=True)  

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        # Generator
        # pred_n_100 = model.Generator(input_n_20)     
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.25, mode='gaussian')
        
        # Discriminator
        real_dec,  real_rec = sliding_window_inference_multi_output(inputs=input_n_100, roi_size=(64, 64), sw_batch_size=8, predictor=model.Discriminator, overlap=0.5, mode='gaussian')
        fake_dec,  fake_rec = sliding_window_inference_multi_output(inputs=pred_n_100, roi_size=(64, 64), sw_batch_size=8, predictor=model.Discriminator, overlap=0.5, mode='gaussian')

        L1_loss      = criterion(pred_n_100, input_n_100)        
        vgg_loss     = compute_Perceptual(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), option=False, device='cuda')
        texture_loss = compute_TML(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), option=False, device='cuda')
        metric_logger.update(L1_loss=L1_loss.item(), VGG_loss=vgg_loss.item(), TML=texture_loss.item())

        # Consistency
        c_real_dec,  _ = sliding_window_inference_multi_output(inputs=real_rec.clip(0, 1), roi_size=(64, 64), sw_batch_size=8, predictor=model.Discriminator, overlap=0.5, mode='gaussian')
        c_fake_dec,  _ = sliding_window_inference_multi_output(inputs=fake_rec.clip(0, 1), roi_size=(64, 64), sw_batch_size=8, predictor=model.Discriminator, overlap=0.5, mode='gaussian')        

    # # Denormalize (No windowing input version)
    # input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    # input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    # pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 
    # # PNG Save
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)
    input_n_100  = fn_tonumpy(input_n_100)
    pred_n_100   = fn_tonumpy(pred_n_100)
    real_dec   = fn_tonumpy(real_dec)
    real_rec   = fn_tonumpy(real_rec)
    fake_dec   = fn_tonumpy(fake_dec)
    fake_rec   = fn_tonumpy(fake_rec)
    
    # Consistency
    c_real_dec   = fn_tonumpy(c_real_dec)    
    c_fake_dec   = fn_tonumpy(c_fake_dec)
    

    # PNG Save
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_real_dec.png', real_dec.squeeze(), cmap="jet")    
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_real_rec.png', real_rec.squeeze(), cmap="gray")    
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_fake_dec.png', fake_dec.squeeze(), cmap="jet")    
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_fake_rec.png', fake_rec.squeeze(), cmap="gray")
    
    # Consistency
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_c_real_dec.png', c_real_dec.squeeze(), cmap="jet")      
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_c_fake_dec.png', c_fake_dec.squeeze(), cmap="jet")    

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_MTD_GAN_Ours(model, criterion, data_loader, device, png_save_dir):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)

    x_features    = []
    y_features    = []
    pred_features = []

    # Metric
    path_list = []
    pl_list   = []
    tml_list  = []
    rmse_list = []
    psnr_list = []
    ssim_list = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        pred_n_100 = model.Generator(input_n_20)     
        # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.25, mode='gaussian').clip(min=0, max=1)
        
        # OLD
        # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.5, mode='constant').clip(min=0, max=1)        

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder

        # # Denormalize (No windowing input version)
        # input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        # input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        # pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
                
        # # Metric
        # original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=4095.0)
        # metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        # metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        # metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # # DCM Save
        # save_dicom(batch_data['path_n_20'][0],  input_n_20,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.dcm'))        
        # save_dicom(batch_data['path_n_100'][0], input_n_100, png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.dcm'))
        # save_dicom(batch_data['path_n_20'][0],  pred_n_100,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.dcm'))        

        # # PNG Save clip for windowing visualize, brain:[0, 80] HU
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.clip(min=0, max=80).squeeze(), cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)

        # Denormalize (windowing input version)

        # Perceptual & FID
        x_feature, y_feature, pred_feature       = compute_feat(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_percep, pred_percep, gt_percep = compute_Perceptual(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_tml, pred_tml, gt_tml          = compute_TML(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        metric_logger.update(input_percep=originial_percep, pred_percep=pred_percep, gt_percep=gt_percep)
        metric_logger.update(input_tml=originial_tml,       pred_tml=pred_tml,       gt_tml=gt_tml)
        x_features.append(x_feature); y_features.append(y_feature); pred_features.append(pred_feature)

        # Metric
        input_n_20, input_n_100, pred_n_100 = fn_tonumpy(input_n_20), fn_tonumpy(input_n_100), fn_tonumpy(pred_n_100)

        original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=1.0)        
        metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")


        # Metric
        path_list.append(batch_data['path_n_20'][0])
        pl_list.append(pred_percep.item())
        tml_list.append(pred_tml.item())
        rmse_list.append(pred_result[2])
        psnr_list.append(pred_result[0])
        ssim_list.append(pred_result[1])

        # # INFERENCE
        # os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[-1], mode=0o777, exist_ok=True) # dicom save folder
        # os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[-1], mode=0o777, exist_ok=True) # png   save folder

        # # Denormalize (windowing input version)
        # input_n_20    = fn_tonumpy(input_n_20)
        # input_n_100   = fn_tonumpy(input_n_100)
        # pred_n_100    = fn_tonumpy(pred_n_100)  

        # # Metric
        # original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=1.0)        
        # metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        # metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        # metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # # PNG Save clip for windowing visualize, brain:[0, 80] HU
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        # plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

    # FID
    originial_fid, pred_fid, gt_fid = compute_FID(torch.cat(x_features, dim=0), torch.cat(y_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=originial_fid, pred_fid=pred_fid, gt_fid=gt_fid)   

    # DataFrame
    df = pd.DataFrame()
    df['PATH'] = path_list
    df['PL'] = pl_list
    df['TML'] = tml_list
    df['RMSE'] = rmse_list
    df['PSNR'] = psnr_list
    df['SSIM'] = ssim_list
    df.to_csv(png_save_dir+'pred_results.csv')
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}






###################################################################             Previous Works                  ###################################################################
# CNN Based  
def train_CNN_Based_Previous(model, data_loader, optimizer, device, epoch, patch_training, print_freq, batch_size):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()
        
        pred_n_100 = model(input_n_20)

        loss = model.criterion(pred_n_100, input_n_100)
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
def valid_CNN_Based_Previous(model, criterion, data_loader, device, epoch, png_save_dir, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid: [epoch:{}]'.format(epoch)
    os.makedirs(png_save_dir, mode=0o777, exist_ok=True) 

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        

        pred_n_100 = model(input_n_20)

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)
 
    # Denormalize (No windowing input version)
    # input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    # input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    # pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 
    # # PNG Save
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)
    input_n_100  = fn_tonumpy(input_n_100)
    pred_n_100   = fn_tonumpy(pred_n_100)
    # PNG Save
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_CNN_Based_Previous(model, criterion, data_loader, device, png_save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)

    x_features    = []
    y_features    = []
    pred_features = []

    # Metric
    path_list = []
    pl_list   = []
    tml_list  = []
    rmse_list = []
    psnr_list = []
    ssim_list = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
                
        pred_n_100 = model(input_n_20).clip(min=0, max=1)

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder
        
        # # Denormalize (No windowing input version)
        # input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        # input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        # pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
                
        # # Metric
        # original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=4095.0)
        # metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        # metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        # metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # # DCM Save
        # save_dicom(batch_data['path_n_20'][0],  input_n_20,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.dcm'))        
        # save_dicom(batch_data['path_n_100'][0], input_n_100, png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.dcm'))
        # save_dicom(batch_data['path_n_20'][0],  pred_n_100,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.dcm'))        

        # # PNG Save clip for windowing visualize, brain:[0, 80] HU
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.clip(min=0, max=80).squeeze(), cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)

        # Denormalize (windowing input version)

        # Perceptual & FID
        x_feature, y_feature, pred_feature       = compute_feat(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_percep, pred_percep, gt_percep = compute_Perceptual(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_tml, pred_tml, gt_tml          = compute_TML(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        metric_logger.update(input_percep=originial_percep, pred_percep=pred_percep, gt_percep=gt_percep)
        metric_logger.update(input_tml=originial_tml,       pred_tml=pred_tml,       gt_tml=gt_tml)
        x_features.append(x_feature); y_features.append(y_feature); pred_features.append(pred_feature)

        # Metric
        input_n_20, input_n_100, pred_n_100 = fn_tonumpy(input_n_20), fn_tonumpy(input_n_100), fn_tonumpy(pred_n_100)

        original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=1.0)        
        metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

        # Metric
        path_list.append(batch_data['path_n_20'][0])
        pl_list.append(pred_percep.item())
        tml_list.append(pred_tml.item())
        rmse_list.append(pred_result[2])
        psnr_list.append(pred_result[0])
        ssim_list.append(pred_result[1])

    # FID
    originial_fid, pred_fid, gt_fid = compute_FID(torch.cat(x_features, dim=0), torch.cat(y_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=originial_fid, pred_fid=pred_fid, gt_fid=gt_fid)   

    # DataFrame
    df = pd.DataFrame()
    df['PATH'] = path_list
    df['PL'] = pl_list
    df['TML'] = tml_list
    df['RMSE'] = rmse_list
    df['PSNR'] = psnr_list
    df['SSIM'] = ssim_list
    df.to_csv(png_save_dir+'pred_results.csv')

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

# Transformer Based  
def train_Transformer_Based_Previous(model, data_loader, optimizer, device, epoch, patch_training, print_freq, batch_size):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()
        
        pred_n_100 = model(input_n_20)

        if model._get_name() == "Restormer":
            loss = model.criterion(pred_n_100, input_n_100)
        elif model._get_name() == "CTformer":
            loss = model.criterion(pred_n_100, input_n_100)*100 + 1e-4  # to prevent 0

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
def valid_Transformer_Based_Previous(model, criterion, data_loader, device, epoch, png_save_dir, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid: [epoch:{}]'.format(epoch)
    os.makedirs(png_save_dir, mode=0o777, exist_ok=True) 

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model, overlap=0.5, mode='constant')     

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)
 
    # Denormalize (No windowing input version)
    # input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    # input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    # pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 
    # # PNG Save
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)
    input_n_100  = fn_tonumpy(input_n_100)
    pred_n_100   = fn_tonumpy(pred_n_100)
    # PNG Save
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_Transformer_Based_Previous(model, criterion, data_loader, device, png_save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)

    x_features    = []
    y_features    = []
    pred_features = []

    # Metric
    path_list = []
    pl_list   = []
    tml_list  = []
    rmse_list = []
    psnr_list = []
    ssim_list = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=16, predictor=model, overlap=0.90, mode='constant')     
        # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model, overlap=0.25, mode='constant')     

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder
    

        # # Denormalize (No windowing input version)
        # input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        # input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        # pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
                
        # # Metric
        # original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=4095.0)
        # metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        # metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        # metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # # DCM Save
        # save_dicom(batch_data['path_n_20'][0],  input_n_20,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.dcm'))        
        # save_dicom(batch_data['path_n_100'][0], input_n_100, png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.dcm'))
        # save_dicom(batch_data['path_n_20'][0],  pred_n_100,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.dcm'))        

        # # PNG Save clip for windowing visualize, brain:[0, 80] HU
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.clip(min=0, max=80).squeeze(), cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)


        # Denormalize (windowing input version)

        # Perceptual & FID
        x_feature, y_feature, pred_feature       = compute_feat(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_percep, pred_percep, gt_percep = compute_Perceptual(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_tml, pred_tml, gt_tml          = compute_TML(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        metric_logger.update(input_percep=originial_percep, pred_percep=pred_percep, gt_percep=gt_percep)
        metric_logger.update(input_tml=originial_tml,       pred_tml=pred_tml,       gt_tml=gt_tml)
        x_features.append(x_feature); y_features.append(y_feature); pred_features.append(pred_feature)

        # Metric
        input_n_20, input_n_100, pred_n_100 = fn_tonumpy(input_n_20), fn_tonumpy(input_n_100), fn_tonumpy(pred_n_100)

        original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=1.0)        
        metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

        # Metric
        path_list.append(batch_data['path_n_20'][0])
        pl_list.append(pred_percep.item())
        tml_list.append(pred_tml.item())
        rmse_list.append(pred_result[2])
        psnr_list.append(pred_result[0])
        ssim_list.append(pred_result[1])

    # FID
    originial_fid, pred_fid, gt_fid = compute_FID(torch.cat(x_features, dim=0), torch.cat(y_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=originial_fid, pred_fid=pred_fid, gt_fid=gt_fid)   

    # DataFrame
    df = pd.DataFrame()
    df['PATH'] = path_list
    df['PL'] = pl_list
    df['TML'] = tml_list
    df['RMSE'] = rmse_list
    df['PSNR'] = psnr_list
    df['SSIM'] = ssim_list
    df.to_csv(png_save_dir+'pred_results.csv')

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}



# GAN Based 
# 1.WGAN
def train_WGAN_VGG_Previous(model, data_loader, optimizer_G, optimizer_D, device, epoch, patch_training, print_freq, batch_size):
    model.Generator.train(True)
    model.Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
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
        g_loss, p_loss = model.g_loss(input_n_20, input_n_100, perceptual=True, return_p=True)
        g_loss.backward()
        optimizer_G.step()
        metric_logger.update(g_loss=g_loss, p_loss=p_loss)

        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_WGAN_VGG_Previous(model, criterion, data_loader, device, epoch, png_save_dir, print_freq, batch_size):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid: [epoch:{}]'.format(epoch)
    os.makedirs(png_save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        pred_n_100 = model.Generator(input_n_20)
        # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.5, mode='constant')     
        # print("input_n_20 == ", input_n_20.shape)
        # print("pred_n_100 == ", pred_n_100.shape)

        L1_loss      = criterion(pred_n_100, input_n_100)        
        vgg_loss     = compute_Perceptual(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), option=False, device='cuda')
        texture_loss = compute_TML(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), option=False, device='cuda')
        metric_logger.update(L1_loss=L1_loss.item(), VGG_loss=vgg_loss.item(), TML=texture_loss.item())

    # Denormalize (No windowing input version)
    # input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    # input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    # pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 
    # # PNG Save
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    # Denormalize (windowing input version)

    # # WGAN-VGG margin value range -> (-40, 120) ---> (0, 1) ---> (-40, 120) ---> (0, 80) ---> (0, 1)
    # input_n_20    = (input_n_20*160.0-40.0).clip(0.0, 80.0) / 80.0
    # input_n_100   = (input_n_100*160.0-40.0).clip(0.0, 80.0) / 80.0
    # pred_n_100    = (pred_n_100.clip(0.0, 1.0)*160.0-40.0).clip(0.0, 80.0) / 80.0

    input_n_20   = fn_tonumpy(input_n_20)
    input_n_100  = fn_tonumpy(input_n_100)
    pred_n_100   = fn_tonumpy(pred_n_100)
    # PNG Save
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_WGAN_VGG_Previous(model, criterion, data_loader, device, png_save_dir):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)    

    x_features    = []
    y_features    = []
    pred_features = []

    # Metric
    path_list = []
    pl_list   = []
    tml_list  = []
    rmse_list = []
    psnr_list = []
    ssim_list = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model.Generator(input_n_20)
        # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.5, mode='constant')     

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder
        
        # # Denormalize (No windowing input version)
        # input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        # input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        # pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
                
        # # Metric
        # original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=4095.0)
        # metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        # metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        # metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # # DCM Save
        # save_dicom(batch_data['path_n_20'][0],  input_n_20,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.dcm'))        
        # save_dicom(batch_data['path_n_100'][0], input_n_100, png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.dcm'))
        # save_dicom(batch_data['path_n_20'][0],  pred_n_100,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.dcm'))        

        # # PNG Save clip for windowing visualize, brain:[0, 80] HU
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.clip(min=0, max=80).squeeze(), cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)

        # Denormalize (windowing input version)

        # # WGAN-VGG margin value range -> (-40, 120) ---> (0, 1) ---> (-40, 120) ---> (0, 80) ---> (0, 1)
        # input_n_20    = (input_n_20*160.0-40.0).clip(0.0, 80.0) / 80.0
        # input_n_100   = (input_n_100*160.0-40.0).clip(0.0, 80.0) / 80.0
        # pred_n_100    = (pred_n_100.clip(0.0, 1.0)*160.0-40.0).clip(0.0, 80.0) / 80.0

        # Perceptual & FID
        x_feature, y_feature, pred_feature       = compute_feat(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_percep, pred_percep, gt_percep = compute_Perceptual(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_tml, pred_tml, gt_tml          = compute_TML(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        metric_logger.update(input_percep=originial_percep, pred_percep=pred_percep, gt_percep=gt_percep)
        metric_logger.update(input_tml=originial_tml,       pred_tml=pred_tml,       gt_tml=gt_tml)
        x_features.append(x_feature); y_features.append(y_feature); pred_features.append(pred_feature)

        # Metric
        input_n_20, input_n_100, pred_n_100 = fn_tonumpy(input_n_20), fn_tonumpy(input_n_100), fn_tonumpy(pred_n_100)

        original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=1.0)        
        metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

        # Metric
        path_list.append(batch_data['path_n_20'][0])
        pl_list.append(pred_percep.item())
        tml_list.append(pred_tml.item())
        rmse_list.append(pred_result[2])
        psnr_list.append(pred_result[0])
        ssim_list.append(pred_result[1])

    # FID
    originial_fid, pred_fid, gt_fid = compute_FID(torch.cat(x_features, dim=0), torch.cat(y_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=originial_fid, pred_fid=pred_fid, gt_fid=gt_fid)   

    # DataFrame
    df = pd.DataFrame()
    df['PATH'] = path_list
    df['PL'] = pl_list
    df['TML'] = tml_list
    df['RMSE'] = rmse_list
    df['PSNR'] = psnr_list
    df['SSIM'] = ssim_list
    df.to_csv(png_save_dir+'pred_results.csv')

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


# 2.MAP_NN
def train_MAP_NN_Previous(model, data_loader, optimizer_G, optimizer_D, device, epoch, patch_training, print_freq, batch_size):
    model.Generator.train(True)
    model.Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
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

        # Generator, perceptual loss
        optimizer_G.zero_grad()
        model.Generator.zero_grad()     # Same as optimizer zero grad()
        g_loss, adv_loss, mse_loss, edge_loss = model.g_loss(input_n_20, input_n_100)
        g_loss.backward()
        optimizer_G.step()
        metric_logger.update(g_loss=g_loss, adv_loss=adv_loss, mse_loss=mse_loss, edge_loss=edge_loss)
        
        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_MAP_NN_Previous(model, criterion, data_loader, device, epoch, png_save_dir, print_freq, batch_size):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid: [epoch:{}]'.format(epoch)
    os.makedirs(png_save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        pred_n_100 = model.Generator(input_n_20)     
            
        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # Denormalize (No windowing input version)
    # input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    # input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    # pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 
    # # PNG Save
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)
    input_n_100  = fn_tonumpy(input_n_100)
    pred_n_100   = fn_tonumpy(pred_n_100)
    # PNG Save
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_MAP_NN_Previous(model, criterion, data_loader, device, png_save_dir):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)    

    x_features    = []
    y_features    = []
    pred_features = []

    # Metric
    path_list = []
    pl_list   = []
    tml_list  = []
    rmse_list = []
    psnr_list = []
    ssim_list = []

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model.Generator(input_n_20)

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder
        
        # # Denormalize (No windowing input version)
        # input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        # input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        # pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
                
        # # Metric
        # original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=4095.0)
        # metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        # metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        # metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # # DCM Save
        # save_dicom(batch_data['path_n_20'][0],  input_n_20,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.dcm'))        
        # save_dicom(batch_data['path_n_100'][0], input_n_100, png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.dcm'))
        # save_dicom(batch_data['path_n_20'][0],  pred_n_100,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.dcm'))        

        # # PNG Save clip for windowing visualize, brain:[0, 80] HU
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.clip(min=0, max=80).squeeze(), cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)

        # Denormalize (windowing input version)

        # Perceptual & FID
        x_feature, y_feature, pred_feature       = compute_feat(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_percep, pred_percep, gt_percep = compute_Perceptual(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_tml, pred_tml, gt_tml          = compute_TML(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        metric_logger.update(input_percep=originial_percep, pred_percep=pred_percep, gt_percep=gt_percep)
        metric_logger.update(input_tml=originial_tml,       pred_tml=pred_tml,       gt_tml=gt_tml)
        x_features.append(x_feature); y_features.append(y_feature); pred_features.append(pred_feature)

        # Metric
        input_n_20, input_n_100, pred_n_100 = fn_tonumpy(input_n_20), fn_tonumpy(input_n_100), fn_tonumpy(pred_n_100)

        original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=1.0)        
        metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

        # Metric
        path_list.append(batch_data['path_n_20'][0])
        pl_list.append(pred_percep.item())
        tml_list.append(pred_tml.item())
        rmse_list.append(pred_result[2])
        psnr_list.append(pred_result[0])
        ssim_list.append(pred_result[1])

    # FID
    originial_fid, pred_fid, gt_fid = compute_FID(torch.cat(x_features, dim=0), torch.cat(y_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=originial_fid, pred_fid=pred_fid, gt_fid=gt_fid)   

    # DataFrame
    df = pd.DataFrame()
    df['PATH'] = path_list
    df['PL'] = pl_list
    df['TML'] = tml_list
    df['RMSE'] = rmse_list
    df['PSNR'] = psnr_list
    df['SSIM'] = ssim_list
    df.to_csv(png_save_dir+'pred_results.csv')

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


# 3.Markovian_Patch_GAN
def train_Markovian_Patch_GAN_Previous(model, data_loader, optimizer_G, optimizer_D, device, epoch, patch_training, print_freq, batch_size):
    model.Generator.train(True)
    model.Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()

        # Discriminator
        optimizer_D.zero_grad()
        model.Discriminator.zero_grad()  
        d_loss = model.d_loss(input_n_20, input_n_100)
        d_loss.backward()
        optimizer_D.step()

        # Generator, perceptual loss
        optimizer_G.zero_grad()
        model.Generator.zero_grad()
        g_loss = model.g_loss(input_n_20, input_n_100)
        g_loss.backward()
        optimizer_G.step()
        
        metric_logger.update(g_loss=g_loss, d_loss=d_loss)
        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"], lr_D=optimizer_D.param_groups[0]["lr"])

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Markovian_Patch_GAN_Previous(model, criterion, data_loader, device, epoch, png_save_dir, print_freq, batch_size):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid: [epoch:{}]'.format(epoch)
    os.makedirs(png_save_dir, mode=0o777, exist_ok=True) 

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # pred_n_100 = model.Generator(input_n_20) # error... caz attention.
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.5, mode='constant')
            
        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)


    # Denormalize (No windowing input version)
    # input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    # input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    # pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 
    # # PNG Save
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)
    input_n_100  = fn_tonumpy(input_n_100)
    pred_n_100   = fn_tonumpy(pred_n_100)
    # PNG Save
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_Markovian_Patch_GAN_Previous(model, criterion, data_loader, device, png_save_dir):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)    

    x_features    = []
    y_features    = []
    pred_features = []    

    # Metric
    path_list = []
    pl_list   = []
    tml_list  = []
    rmse_list = []
    psnr_list = []
    ssim_list = []

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model.Generator(input_n_20)

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder
        
        # # Denormalize (No windowing input version)
        # input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        # input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        # pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
                
        # # Metric
        # original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=4095.0)
        # metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        # metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        # metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # # DCM Save
        # save_dicom(batch_data['path_n_20'][0],  input_n_20,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.dcm'))        
        # save_dicom(batch_data['path_n_100'][0], input_n_100, png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.dcm'))
        # save_dicom(batch_data['path_n_20'][0],  pred_n_100,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.dcm'))        

        # # PNG Save clip for windowing visualize, brain:[0, 80] HU
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.clip(min=0, max=80).squeeze(), cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)

        # Denormalize (windowing input version)

        # Perceptual & FID
        x_feature, y_feature, pred_feature       = compute_feat(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_percep, pred_percep, gt_percep = compute_Perceptual(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_tml, pred_tml, gt_tml          = compute_TML(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        metric_logger.update(input_percep=originial_percep, pred_percep=pred_percep, gt_percep=gt_percep)
        metric_logger.update(input_tml=originial_tml,       pred_tml=pred_tml,       gt_tml=gt_tml)
        x_features.append(x_feature); y_features.append(y_feature); pred_features.append(pred_feature)

        # Metric
        input_n_20, input_n_100, pred_n_100 = fn_tonumpy(input_n_20), fn_tonumpy(input_n_100), fn_tonumpy(pred_n_100)

        original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=1.0)        
        metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

        # Metric
        path_list.append(batch_data['path_n_20'][0])
        pl_list.append(pred_percep.item())
        tml_list.append(pred_tml.item())
        rmse_list.append(pred_result[2])
        psnr_list.append(pred_result[0])
        ssim_list.append(pred_result[1])

    # FID
    originial_fid, pred_fid, gt_fid = compute_FID(torch.cat(x_features, dim=0), torch.cat(y_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=originial_fid, pred_fid=pred_fid, gt_fid=gt_fid)   

    # DataFrame
    df = pd.DataFrame()
    df['PATH'] = path_list
    df['PL'] = pl_list
    df['TML'] = tml_list
    df['RMSE'] = rmse_list
    df['PSNR'] = psnr_list
    df['SSIM'] = ssim_list
    df.to_csv(png_save_dir+'pred_results.csv')

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}



# 4.DUGAN
def train_DUGAN_Previous(model, data_loader, optimizer_G, optimizer_Img_D, optimizer_Grad_D, device, epoch, patch_training, print_freq, batch_size):
    model.Generator.train(True)
    model.Image_Discriminator.train(True)
    model.Grad_Discriminator.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
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
def valid_DUGAN_Previous(model, criterion, data_loader, device, epoch, png_save_dir, print_freq, batch_size):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid: [epoch:{}]'.format(epoch)
    os.makedirs(png_save_dir, mode=0o777, exist_ok=True)  

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()

        pred_n_100 = model.Generator(input_n_20)     
            
        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)


    # Denormalize (No windowing input version)
    # input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    # input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    # pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 
    # # PNG Save
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    # Denormalize (windowing input version)
    input_n_20   = fn_tonumpy(input_n_20)
    input_n_100  = fn_tonumpy(input_n_100)
    pred_n_100   = fn_tonumpy(pred_n_100)
    # PNG Save
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray")
    plt.imsave(png_save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_DUGAN_Previous(model, criterion, data_loader, device, png_save_dir):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)    
    
    x_features    = []
    y_features    = []
    pred_features = []

    # Metric
    path_list = []
    pl_list   = []
    tml_list  = []
    rmse_list = []
    psnr_list = []
    ssim_list = []

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=10):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model.Generator(input_n_20)

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)            

        # SAVE
        os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder
        
        # # Denormalize (No windowing input version)
        # input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        # input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        # pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
                
        # # Metric
        # original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=4095.0)
        # metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        # metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        # metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # # DCM Save
        # save_dicom(batch_data['path_n_20'][0],  input_n_20,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.dcm'))        
        # save_dicom(batch_data['path_n_100'][0], input_n_100, png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.dcm'))
        # save_dicom(batch_data['path_n_20'][0],  pred_n_100,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.dcm'))        

        # # PNG Save clip for windowing visualize, brain:[0, 80] HU
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.clip(min=0, max=80).squeeze(), cmap="gray", vmin=0, vmax=80)
        # plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.clip(min=0, max=80).squeeze(),  cmap="gray", vmin=0, vmax=80)

        # Denormalize (windowing input version)

        # Perceptual & FID
        x_feature, y_feature, pred_feature       = compute_feat(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_percep, pred_percep, gt_percep = compute_Perceptual(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        originial_tml, pred_tml, gt_tml          = compute_TML(x=input_n_20, y=input_n_100, pred=pred_n_100.clip(0, 1), device='cuda')
        metric_logger.update(input_percep=originial_percep, pred_percep=pred_percep, gt_percep=gt_percep)
        metric_logger.update(input_tml=originial_tml,       pred_tml=pred_tml,       gt_tml=gt_tml)
        x_features.append(x_feature); y_features.append(y_feature); pred_features.append(pred_feature)

        # Metric
        input_n_20, input_n_100, pred_n_100 = fn_tonumpy(input_n_20), fn_tonumpy(input_n_100), fn_tonumpy(pred_n_100)

        original_result, pred_result, gt_result = compute_measure(x=torch.tensor(input_n_20).squeeze(), y=torch.tensor(input_n_100).squeeze(), pred=torch.tensor(pred_n_100).squeeze(), data_range=1.0)        
        metric_logger.update(input_psnr=original_result[0], input_ssim=original_result[1], input_rmse=original_result[2])   
        metric_logger.update(pred_psnr=pred_result[0],      pred_ssim=pred_result[1],      pred_rmse=pred_result[2])   
        metric_logger.update(gt_psnr=gt_result[0],          gt_ssim=gt_result[1],          gt_rmse=gt_result[2])   

        # PNG Save clip for windowing visualize, brain:[0, 80] HU
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray")
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray")

        # Metric
        path_list.append(batch_data['path_n_20'][0])
        pl_list.append(pred_percep.item())
        tml_list.append(pred_tml.item())
        rmse_list.append(pred_result[2])
        psnr_list.append(pred_result[0])
        ssim_list.append(pred_result[1])

    # FID
    originial_fid, pred_fid, gt_fid = compute_FID(torch.cat(x_features, dim=0), torch.cat(y_features, dim=0), torch.cat(pred_features, dim=0))
    metric_logger.update(input_fid=originial_fid, pred_fid=pred_fid, gt_fid=gt_fid)   

    # DataFrame
    df = pd.DataFrame()
    df['PATH'] = path_list
    df['PL'] = pl_list
    df['TML'] = tml_list
    df['RMSE'] = rmse_list
    df['PSNR'] = psnr_list
    df['SSIM'] = ssim_list
    df.to_csv(png_save_dir+'pred_results.csv')

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

