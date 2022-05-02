from fcntl import DN_DELETE
import math
import utils
import torch
from torch.nn import functional as F

import numpy as np
from pydicom import dcmread
from tqdm import tqdm
import sys
import os 
import matplotlib.pyplot as plt
from metrics import compute_measure, compute_measure_3D
from monai.inferers import sliding_window_inference
from module.sliding_window_inference_SACNN import sliding_window_inference_sacnn


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
def train_CNN_Based_Ours(model, criterion, data_loader, optimizer, device, epoch, patch_training, loss_name):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)
    print_freq = 10  
    loss_detail = None

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()
        

        if model._get_name() == "ResFFT_Freq_SPADE_Att_window":
            input_n_20  = input_n_20.clip(0.250,  0.270)
            input_n_20 -= input_n_20.min()
            input_n_20 /= input_n_20.max()

            input_n_100 = input_n_100.clip(0.250,  0.270)
            input_n_100 -= input_n_100.min()
            input_n_100 /= input_n_100.max()

        pred_n_100 = model(input_n_20)
        # print("Check = ", pred_n_100[0].max(), pred_n_100[0].min(), pred_n_100[0].dtype, pred_n_100[0].shape)
        # print("Check = ", input_n_100.max(), input_n_100.min(), input_n_100.dtype, input_n_100.shape) # [32, 1, 64, 64]
        if loss_name == 'Change L2 L1 Loss':
            loss = criterion(pred_n_100, input_n_100, epoch)

        elif loss_name == 'Perceptual_Triple+L1_Loss':    
            loss = criterion(gt_low=input_n_20, gt_high=input_n_100, target=pred_n_100)            

        elif loss_name == 'Window L1 Loss':    
            loss = criterion(gt_high=input_n_100, target=pred_n_100)       

        elif loss_name == 'Perceptual+L1 Loss':    
            loss = criterion(gt_100=input_n_100, pred_n_100=pred_n_100)       

        elif loss_name == 'Charbonnier_HighFreq_Loss':    
            loss, loss_detail = criterion(gt_100=input_n_100, pred_n_100=pred_n_100)       

        elif loss_name == 'Charbonnier_Edge_MSFR_Loss':    
            loss, loss_detail = criterion(gt_100=input_n_100, pred_n_100=pred_n_100)           

        elif loss_name == 'Charbonnier_Edge_MSFR_VGG_Loss':    
            loss, loss_detail = criterion(gt_100=input_n_100, pred_n_100=pred_n_100)                               

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
def valid_CNN_Based_Ours(model, criterion, data_loader, device, epoch, save_dir, loss_name):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    
    loss_detail = None

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        if model._get_name() == "ResFFT_Freq_SPADE_Att_window":
            input_n_20  = input_n_20.clip(0.250,  0.270)
            input_n_20 -= input_n_20.min()
            input_n_20 /= input_n_20.max()

            input_n_100 = input_n_100.clip(0.250,  0.270)
            input_n_100 -= input_n_100.min()
            input_n_100 /= input_n_100.max()

        if hasattr(model, 'module'):
            if model.module._get_name() == "SPADE_UNet" or model.module._get_name() == "SPADE_UNet_Upgrade" or model.module._get_name() == "ResFFT_LFSPADE" or model.module._get_name() == 'ResFFT_Freq_SPADE_Att' or model.module._get_name() == 'ResFFT_Freq_SPADE_Att_window':
                pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.module, overlap=0.5, mode='constant')
            else:
                pred_n_100 = model(input_n_20)

        else :
            if model._get_name() == "SPADE_UNet" or model._get_name() == "SPADE_UNet_Upgrade" or model._get_name() == "ResFFT_LFSPADE" or model._get_name() == 'ResFFT_Freq_SPADE_Att' or model._get_name() == 'ResFFT_Freq_SPADE_Att_window':
                pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model, overlap=0.5, mode='constant')     
            else:
                pred_n_100 = model(input_n_20)


        if loss_name == 'Change L2 L1 Loss':
            loss = criterion(pred_n_100, input_n_100, epoch)

        elif loss_name == 'Perceptual_Triple+L1_Loss':    
            loss = criterion(gt_low=input_n_20, gt_high=input_n_100, target=pred_n_100)            

        elif loss_name == 'Window L1 Loss':    
            loss = criterion(gt_high=input_n_100, target=pred_n_100)    

        elif loss_name == 'Perceptual+L1 Loss':    
            loss = criterion(gt_100=input_n_100, pred_n_100=pred_n_100)       

        elif loss_name == 'Charbonnier_HighFreq_Loss':    
            loss, loss_detail = criterion(gt_100=input_n_100, pred_n_100=pred_n_100) 

        elif loss_name == 'Charbonnier_Edge_MSFR_Loss':    
            loss, loss_detail = criterion(gt_100=input_n_100, pred_n_100=pred_n_100) 

        elif loss_name == 'Charbonnier_Edge_MSFR_VGG_Loss':    
            loss, loss_detail = criterion(gt_100=input_n_100, pred_n_100=pred_n_100)             

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
 
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    if model._get_name() == "ResFFT_Freq_SPADE_Att_window":
        # PNG Save
        input_n_20   = fn_tonumpy(input_n_20)
        input_n_100  = fn_tonumpy(input_n_100)
        pred_n_100   = fn_tonumpy(pred_n_100)

        print(save_dir+'epoch_'+str(epoch)+'_input_n_20.png')    
        plt.imsave(save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
        plt.imsave(save_dir+'epoch_'+str(epoch)+'_gt_n_100.png', input_n_100.squeeze(), cmap="gray")
        plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")

    else: 
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
# 1. FSGAN
def train_FSGAN_Previous(model, data_loader, optimizer_G, optimizer_Low_D, optimizer_High_D, device, epoch, patch_training):
    model.Generator.train(True)
    model.Low_discriminator.train(True)
    model.High_discriminator.train(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)
    print_freq = 10  

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()

        # Inference Setting
        low_dose                      = input_n_20
        full_dose                     = input_n_100
        # mu, logvar, gen_full_dose     = model.Generator(low_dose)
        gen_full_dose     = model.Generator(low_dose)

        # Low Discriminator
        optimizer_Low_D.zero_grad()
        model.Low_discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(1):
            Low_D_loss = model.train_Low_Discriminator(full_dose, low_dose, gen_full_dose, prefix='Low_Freq', n_iter=epoch)
            Low_D_loss.backward()
            optimizer_Low_D.step()
        
        # High Discriminator
        optimizer_High_D.zero_grad()
        model.High_discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(1):
            High_D_loss = model.train_High_Discriminator(full_dose, low_dose, gen_full_dose, prefix='High_Freq', n_iter=epoch)
            High_D_loss.backward()
            optimizer_High_D.step()        

        # Generator
        optimizer_G.zero_grad()
        model.Generator.zero_grad()     # Same as optimizer zero grad()
        
            # Low
        low_gen_enc, low_gen_dec   = model.Low_discriminator(gen_full_dose)
        low_gen_loss               = model.gan_metric(low_gen_enc, torch.ones_like(low_gen_enc)) + model.gan_metric(low_gen_dec, torch.ones_like(low_gen_dec))
            # High
        high_gen_enc, high_gen_dec = model.High_discriminator(gen_full_dose)
        high_gen_loss              = model.gan_metric(high_gen_enc, torch.ones_like(high_gen_enc)) + model.gan_metric(high_gen_dec, torch.ones_like(high_gen_dec))

        # adv_loss  = 0.1*low_gen_loss + 0.1*high_gen_loss 
        # pix_loss  = 1.0*F.l1_loss(gen_full_dose, full_dose)         
        # enc_loss  = 0.05*model.KLDLoss(mu, logvar)
        # G_loss = adv_loss + pix_loss + enc_loss

        adv_loss  = 0.1*low_gen_loss + 0.1*high_gen_loss 
        pix_loss  = 1000.0*model.pixel_metric(gen_full_dose, full_dose)         
        G_loss = adv_loss + pix_loss
                 
        G_loss.backward()        
        optimizer_G.step()


        G_dict = {}
        G_dict.update({
            'G_loss/low_loss': low_gen_loss,
            'G_loss/high_loss': high_gen_loss,
            'G_loss/pix_loss': pix_loss,
            # 'G_loss/enc_loss': enc_loss,

            'D_loss/low_loss': Low_D_loss.item(),    
            'D_loss/high_loss': High_D_loss.item(),
        })

        metric_logger.update(**G_dict)
        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_FSGAN_Previous(model, criterion, data_loader, device, epoch, save_dir):
    model.Generator.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
    

        if hasattr(model, 'module'):
            if model.module._get_name() == "FSGAN":
                pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.module.Generator.inference, overlap=0.5, mode='constant')
            else:
                pred_n_100 = model.Generator(input_n_20)     

        else :
            if model._get_name() == "FSGAN":
                pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator.inference, overlap=0.5, mode='constant')     
            else:
                pred_n_100 = model.Generator(input_n_20)     

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    # Denormalize
    input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 

    # PNG Save
    print(save_dir+'epoch_'+str(epoch)+'_input_n_20.png')
    
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_gt_n_100.png', input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_FSGAN_Previous(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.Generator.eval()
    
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
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator.inference, overlap=0.5, mode='constant')

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


# 2. FDGAN
def train_FDGAN_Previous(model, data_loader, optimizer_G, optimizer_Img_D, optimizer_Fourier_D, device, epoch, patch_training):
    model.Generator.train(True)
    model.Img_discriminator.train(True)
    model.Fourier_discriminator.train(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)
    print_freq = 1  

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()

        # Inference Setting
        low_dose                      = input_n_20
        full_dose                     = input_n_100
        # mu, logvar, gen_full_dose     = model.Generator(low_dose)
        gen_full_dose     = model.Generator(low_dose)

        # Img Discriminator
        optimizer_Img_D.zero_grad()
        model.Img_discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(1):
            Img_D_loss = model.train_Img_Discriminator(full_dose, low_dose, gen_full_dose, prefix='Img_D', n_iter=epoch)
            Img_D_loss.backward()
            optimizer_Img_D.step()
        
        # Fourier Discriminator
        optimizer_Fourier_D.zero_grad()
        model.Fourier_discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(1):
            Fourier_D_loss = model.train_Fourier_Discriminator(full_dose, low_dose, gen_full_dose, prefix='Fourier_D', n_iter=epoch)
            Fourier_D_loss.backward()
            optimizer_Fourier_D.step()        

        # Generator
        optimizer_G.zero_grad()
        model.Generator.zero_grad()     # Same as optimizer zero grad()
        
            # Low
        img_gen_enc, img_gen_dec   = model.Img_discriminator(gen_full_dose)
        img_gen_loss               = model.gan_metric(img_gen_enc, torch.ones_like(img_gen_enc)) + model.gan_metric(img_gen_dec, torch.ones_like(img_gen_dec))
            # High
        fourier_gen_enc, fourier_gen_dec = model.Fourier_discriminator(gen_full_dose)
        fourier_gen_loss                 = model.gan_metric(fourier_gen_enc, torch.ones_like(fourier_gen_enc)) + model.gan_metric(fourier_gen_dec, torch.ones_like(fourier_gen_dec))

        # adv_loss  = 0.1*low_gen_loss + 0.1*high_gen_loss 
        # pix_loss  = 1.0*F.l1_loss(gen_full_dose, full_dose)         
        # enc_loss  = 0.05*model.KLDLoss(mu, logvar)
        # G_loss = adv_loss + pix_loss + enc_loss

        adv_loss   = 0.1*img_gen_loss + 0.1*fourier_gen_loss 
        pix_loss1  = 1.0*model.pixel_metric1(gen_full_dose, full_dose) 
        pix_loss2  = 0.5*model.pixel_metric2(gen_full_dose, full_dose)
        pix_loss3  = 0.5*model.pixel_metric3(gen_full_dose, full_dose)        

        G_loss    = adv_loss + pix_loss1 + pix_loss2 + pix_loss3
                 
        G_loss.backward()        
        optimizer_G.step()


        G_dict = {}
        G_dict.update({
            'G_loss/img_gen_loss': img_gen_loss,
            'G_loss/fourier_gen_loss': fourier_gen_loss,
            'G_loss/pix_loss1': pix_loss1,
            'G_loss/pix_loss2': pix_loss2,
            'G_loss/pix_loss3': pix_loss3,
            # 'G_loss/enc_loss': enc_loss,

            'D_loss/Img_D_loss': Img_D_loss.item(),    
            'D_loss/Fourier_D_loss': Fourier_D_loss.item(),
        })

        metric_logger.update(**G_dict)
        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_FDGAN_Previous(model, criterion, data_loader, device, epoch, save_dir):
    model.Generator.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
    
        if hasattr(model, 'module'):
            if model.module._get_name() == "FDGAN":
                pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.module.Generator, overlap=0.5, mode='constant')
            else:
                pred_n_100 = model.Generator(input_n_20)     

        else :
            if model._get_name() == "FDGAN":
                pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.5, mode='constant')     
            else:
                pred_n_100 = model.Generator(input_n_20)     

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    # Denormalize
    input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 

    # PNG Save
    print(save_dir+'epoch_'+str(epoch)+'_input_n_20.png')
    
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_gt_n_100.png', input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_FDGAN_Previous(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.Generator.eval()
    
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
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator.inference, overlap=0.5, mode='constant')

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



# 3. FDGAN_PatchGAN
def train_FDGAN_PatchGAN_Ours(model, data_loader, optimizer_G, optimizer_Image_D, optimizer_Fourier_D, device, epoch, patch_training):
    model.Generator.train(True)
    model.Image_discriminator.train(True)
    model.Fourier_discriminator.train(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)
    print_freq = 1  

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).to(device).float()  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].to(device).float()
            input_n_100  = batch_data['n_100'].to(device).float()

        # Inference Setting
        low_dose       = input_n_20
        full_dose      = input_n_100
        gen_full_dose  = model.Generator(low_dose)

        # Img Discriminator
        optimizer_Image_D.zero_grad()
        model.Image_discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(1):
            Img_D_loss = model.train_Image_Discriminator(full_dose, low_dose, gen_full_dose, prefix='Image_D', n_iter=epoch)
            Img_D_loss.backward()
            optimizer_Image_D.step()
        
        # Fourier Discriminator
        optimizer_Fourier_D.zero_grad()
        model.Fourier_discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(1):
            Fourier_D_loss = model.train_Fourier_Discriminator(full_dose, low_dose, gen_full_dose, prefix='Fourier_D', n_iter=epoch)
            Fourier_D_loss.backward()
            optimizer_Fourier_D.step()        

        # Generator
        optimizer_G.zero_grad()
        model.Generator.zero_grad()     # Same as optimizer zero grad()
        
            # Low
        image_gen          = model.Image_discriminator(gen_full_dose)
        # image_gen_loss     = model.gan_metric(image_gen[0], torch.ones_like(image_gen[0]))
        image_gen_loss     = model.gan_metric(image_gen, torch.ones_like(image_gen))
        
            # High
        fourier_gen        = model.Fourier_discriminator(gen_full_dose)
        fourier_gen_loss   = model.gan_metric(fourier_gen, torch.ones_like(fourier_gen)) 


        adv_loss   = 0.1*image_gen_loss + 0.1*fourier_gen_loss 
        pix_loss1  = 1.0*model.pixel_metric1(gen_full_dose, full_dose) 
        pix_loss2  = 0.5*model.pixel_metric2(gen_full_dose, full_dose)
        pix_loss3  = 0.5*model.pixel_metric3(gen_full_dose, full_dose)        

        G_loss    = adv_loss + pix_loss1 + pix_loss2 + pix_loss3
                 
        G_loss.backward()        
        optimizer_G.step()


        G_dict = {}
        G_dict.update({
            'G_loss/image_gen_loss': image_gen_loss,
            'G_loss/fourier_gen_loss': fourier_gen_loss,
            'G_loss/pix_loss1': pix_loss1,
            'G_loss/pix_loss2': pix_loss2,
            'G_loss/pix_loss3': pix_loss3,

            'D_loss/Image_D_loss': Img_D_loss.item(),    
            'D_loss/Fourier_D_loss': Fourier_D_loss.item(),
        })

        metric_logger.update(**G_dict)
        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_FDGAN_PatchGAN_Ours(model, criterion, data_loader, device, epoch, save_dir):
    # model.module.Generator.eval() if hasattr(model, 'module') else model.Generator.eval()
    model.Generator.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()


        if hasattr(model, 'module'):
            # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.module.Generator, overlap=0.5, mode='constant')
            pred_n_100 = model.Generator(input_n_20)     

        else :
            # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.5, mode='constant')     
            pred_n_100 = model.Generator(input_n_20)     

        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    # Denormalize
    input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 

    # PNG Save
    print(save_dir+'epoch_'+str(epoch)+'_input_n_20.png')
    
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_gt_n_100.png', input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_FDGAN_PatchGAN_Ours(model, data_loader, device, save_dir):
    # switch to evaluation mode
    # model.module.Generator.eval() if hasattr(model, 'module') else model.Generator.eval()
    model.Generator.eval()
    
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
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator.inference, overlap=0.5, mode='constant')

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










###################################################################             Previous Works                  ###################################################################
# CNN Based  ################################################

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
def test_CNN_Based_Previous(model, data_loader, device, save_dir):
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
        if model._get_name() == "Restormer" or model._get_name() == "TED_Net" or model._get_name() == "CTformer":
            pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model, overlap=0.5, mode='constant')     
        else:
            pred_n_100 = model(input_n_20)

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



# Transformer Based  ################################################
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
        elif model._get_name() == "TED_Net" or model._get_name() == "CTformer":
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
def test_Transformer_Based_Previous(model, data_loader, device, save_dir):
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






# GAN Based  ################################################
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
def test_WGAN_VGG_Previous(model, data_loader, device, png_save_dir):
    # switch to evaluation mode
    model.Generator.eval()
    
    # compute PSNR, SSIM, RMSE
    ori_psnr_avg,  ori_ssim_avg,  ori_rmse_avg  = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    gt_psnr_avg,   gt_ssim_avg,   gt_rmse_avg   = 0, 0, 0

    for batch_data in tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50):
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model.Generator(input_n_20)
        # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator.inference, overlap=0.5, mode='constant')

        os.makedirs(png_save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(png_save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder
        
        input_n_20    = dicom_denormalize(fn_tonumpy(input_n_20))
        input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
        
        # DCM Save
        save_dicom(batch_data['path_n_20'][0],  input_n_20,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.dcm'))        
        save_dicom(batch_data['path_n_100'][0], input_n_100, png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/' + batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.dcm'))
        save_dicom(batch_data['path_n_20'][0],  pred_n_100,  png_save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]  + '/' + batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.dcm'))        
        
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
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray", vmin=0, vmax=80)
        plt.imsave(png_save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
        plt.imsave(png_save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray", vmin=0, vmax=80)


    print('\n')
    print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(data_loader), ori_ssim_avg/len(data_loader), ori_rmse_avg/len(data_loader)))
    print('\n')
    print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(data_loader), pred_ssim_avg/len(data_loader), pred_rmse_avg/len(data_loader)))        
    print('\n')
    print('GT === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(gt_psnr_avg/len(data_loader), gt_ssim_avg/len(data_loader), gt_rmse_avg/len(data_loader)))        

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
def test_MAP_NN_Previous(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.Generator.eval()
    
    # compute PSNR, SSIM, RMSE
    ori_psnr_avg,  ori_ssim_avg,  ori_rmse_avg  = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    gt_psnr_avg,   gt_ssim_avg,   gt_rmse_avg   = 0, 0, 0

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model.Generator(input_n_20) 
        # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator.inference, overlap=0.5, mode='constant')

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



# 5.SACNN - No
# from monai.transforms import SaveImage
def train_SACNN_Previous_3D(model, data_loader, optimizer_G, optimizer_D, device, epoch, patch_training):
    model.Generator.train(True)
    model.Discriminator.train(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)
    print_freq = 10  

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            f  = torch.cat([ batch_data[i]['n_20_f']  for i in range(8) ])  # 8 is patch_nums
            m  = torch.cat([ batch_data[i]['n_20_m']  for i in range(8) ])  # 8 is patch_nums
            l  = torch.cat([ batch_data[i]['n_20_l']  for i in range(8) ])  # 8 is patch_nums
            input_n_20  = torch.stack([f, m, l], dim=2).to(device).float()
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()
        else :
            f  = batch_data['n_20_f']  # 8 is patch_nums
            m  = batch_data['n_20_m']  # 8 is patch_nums
            l  = batch_data['n_20_l']  # 8 is patch_nums
            input_n_20  = torch.stack([f, m, l], dim=2).to(device).float()
            input_n_100 = batch_data['n_100'].to(device).float()

        # Discriminator, 4 time more training than Generator
        optimizer_D.zero_grad()
        model.Discriminator.zero_grad()
        for _ in range(4):
            d_loss, gp_loss = model.d_loss(input_n_20, input_n_100, gp=True, return_gp=True)
            d_loss.backward()
            optimizer_D.step()

        # Generator, perceptual loss
        optimizer_G.zero_grad()
        model.Generator.zero_grad()
        g_loss, p_loss = model.g_loss(input_n_20, input_n_100, perceptual=True, return_p=True)
        g_loss.backward()
        optimizer_G.step()
        
        metric_logger.update(g_loss=g_loss, d_loss=d_loss, p_loss=p_loss, gp_loss=gp_loss)
        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_SACNN_Previous_3D(model, criterion, data_loader, device, epoch, save_dir):
    model.Generator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        # forward pass
        f   = batch_data['n_20_f']
        m   = batch_data['n_20_m']
        l   = batch_data['n_20_l']

        input_n_20   = torch.stack([f, m, l], dim=2).to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
                     
        # Forward Generator
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(3, 64, 64), sw_batch_size=1, predictor=model.Generator, padding_mode='constant', overlap=0.25, mode='gaussian')
        pred_n_100 = pred_n_100[:,:,1,:,:]

        L1_loss = criterion(pred_n_100, input_n_100)

        loss_value = L1_loss.item()

        metric_logger.update(L1_loss=loss_value)
    
    # np.save('/workspace/sunggu/c.npy', pred_n_100.cpu().detach().numpy()) # 이상없음
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    # Denormalize
    input_n_20   = input_n_20[:,:,1,:,:]
    input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 

    # PNG Save
    print(save_dir+'epoch_'+str(epoch)+'_input_n_20.png')
    
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_SACNN_Previous_3D(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.Generator.eval()
    cnt = 0
    # compute PSNR, SSIM, RMSE
    ori_psnr_avg,  ori_ssim_avg,  ori_rmse_avg  = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    gt_psnr_avg,   gt_ssim_avg,   gt_rmse_avg   = 0, 0, 0

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:
        
        # forward pass
        f   = batch_data['n_20_f']
        m   = batch_data['n_20_m']
        l   = batch_data['n_20_l']

        input_n_20   = torch.stack([f, m, l], dim=2).to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        # print("check1 == ", f.shape)   torch.Size([1, 1, 512, 512])
        # print("check2 == ", input_n_20.shape) torch.Size([1, 1, 3, 512, 512])

        # Forward Generator
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(3, 64, 64), sw_batch_size=1, predictor=model.Generator, padding_mode='constant', overlap=0.25, mode='gaussian')
        pred_n_100 = pred_n_100[:,:,1,:,:]

        os.makedirs(save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder
        
        input_n_20    = input_n_20[:,:,1,:,:]
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
def test_Markovian_Patch_GAN_Previous(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.Generator.eval()
    
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
        pred_n_100 = model.Generator(input_n_20)
        # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator.inference, overlap=0.5, mode='constant')

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
        metric_logger.update(**Img_detail)
            # Grad D
        optimizer_Grad_D.zero_grad()
        model.Grad_Discriminator.zero_grad()        
        Grad_d_loss, Grad_detail = model.Grad_d_loss(input_n_20, input_n_100)
        Grad_d_loss.backward()
        optimizer_Grad_D.step()        
        metric_logger.update(**Grad_detail)

        # Generator
        optimizer_G.zero_grad()
        model.Generator.zero_grad()
        g_loss, g_detail = model.g_loss(input_n_20, input_n_100)
        g_loss.backward()        
        optimizer_G.step()
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
def test_DUGAN_Previous(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.Generator.eval()
    
    # compute PSNR, SSIM, RMSE
    ori_psnr_avg,  ori_ssim_avg,  ori_rmse_avg  = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    gt_psnr_avg,   gt_ssim_avg,   gt_rmse_avg   = 0, 0, 0

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model.Generator(input_n_20)     
        # pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.Generator.inference, overlap=0.5, mode='constant')

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








# ETC      ################################################
# 1. SACNN AutoEncoder
def train_SACNN_AE_Previous_3D(model, criterion, data_loader, optimizer, device, epoch, patch_training):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)
    print_freq = 10  

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).to(device).float()  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_100  = batch_data['n_100'].to(device).float()
        
        # print("Check = ", input_n_20.max(), input_n_20.min(), input_n_20.dtype, input_n_20.shape)
        
        pred_n_100 = model(input_n_100)    

        loss = criterion(pred_n_100, input_n_100)

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
def valid_SACNN_AE_Previous_3D(model, criterion, data_loader, device, epoch, save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # print(input_n_20.shape) # (1, 1, 512, 512)
        
        pred_n_100 = model(input_n_100)     
        # pred_n_100 = sliding_window_inference(inputs=input_n_100, roi_size=(3, 64, 64), sw_batch_size=1, predictor=model, overlap=0.25, mode='gaussian')
        
        loss = criterion(pred_n_100, input_n_100)

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
    input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80) 

    print(save_dir+'epoch_'+str(epoch)+'_input_n_20.png')    
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_gt_n_100.png', input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_SACNN_AE_Previous_3D(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.eval()

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:

        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        pred_n_100 = model(input_n_100)

        os.makedirs(save_dir.replace('/png/', '/dcm/') + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # dicom save folder
        os.makedirs(save_dir                           + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder

        input_n_100   = dicom_denormalize(fn_tonumpy(input_n_100))
        pred_n_100    = dicom_denormalize(fn_tonumpy(pred_n_100))       
        
        # DCM Save       
        save_dicom(batch_data['path_n_100'][0], input_n_100, save_dir.replace('/png/', '/dcm/')+batch_data['path_n_100'][0].split('/')[7] + '/gt_n_100_'    + batch_data['path_n_100'][0].split('/')[-1])
        save_dicom(batch_data['path_n_20'][0],  pred_n_100,  save_dir.replace('/png/', '/dcm/')+batch_data['path_n_20'][0].split('/')[7]   + '/pred_n_100_'  + batch_data['path_n_20'][0].split('/')[-1])        
        
        # PNG Save clip for windowing visualize
        input_n_100   = input_n_100.clip(min=0, max=80)
        pred_n_100    = pred_n_100.clip(min=0, max=80)
        plt.imsave(save_dir+batch_data['path_n_100'][0].split('/')[7]+'/gt_n_100_'  +batch_data['path_n_100'][0].split('/')[-1].replace('.dcm', '.png'), input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
        plt.imsave(save_dir+batch_data['path_n_20'][0].split('/')[7] +'/pred_n_100_'+batch_data['path_n_20'][0].split('/')[-1].replace('.dcm', '.png'),  pred_n_100.squeeze(),  cmap="gray", vmin=0, vmax=80)



# TEST      ################################################

@torch.no_grad()
def test_Unet_GAN_Ours(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.Generator.eval()
    
    # compute PSNR, SSIM, RMSE
    ori_psnr_avg,  ori_ssim_avg,  ori_rmse_avg  = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:
        
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # Forward Generator
        # pred_n_100 = model(input_n_20)
        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.inference, overlap=0.5, mode='constant')

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
        plt.imsave(save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_gt_n_20.png'),     input_n_20.squeeze(),  cmap="gray", vmin=0, vmax=80)
        plt.imsave(save_dir+batch_data['path_n_100'][0].split('/')[7] +'/'+batch_data['path_n_100'][0].split('_')[-1].replace('.dcm', '_gt_n_100.png'),   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
        plt.imsave(save_dir+batch_data['path_n_20'][0].split('/')[7]  +'/'+batch_data['path_n_20'][0].split('_')[-1].replace('.dcm', '_pred_n_100.png'),  pred_n_100.squeeze(),  cmap="gray", vmin=0, vmax=80)

    print('\n')
    print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(data_loader), ori_ssim_avg/len(data_loader), ori_rmse_avg/len(data_loader)))
    print('\n')
    print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(data_loader), pred_ssim_avg/len(data_loader), pred_rmse_avg/len(data_loader)))        
