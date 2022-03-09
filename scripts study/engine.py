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
from sliding_window_inference_SACNN import sliding_window_inference_sacnn




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
        
        if hasattr(model, 'module'):
            if model.module._get_name() == "SPADE_UNet" or model.module._get_name() == "SPADE_UNet_Upgrade":
                pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.module, overlap=0.5, mode='constant')
            else:
                pred_n_100 = model(input_n_20)

        else :
            if model._get_name() == "SPADE_UNet" or model.module._get_name() == "SPADE_UNet_Upgrade":
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




















###################################################################             Previous Works                  ###################################################################
# CNN Based  ################################################
# Train 
def train_CNN_Based_Previous(model, criterion, data_loader, optimizer, device, epoch, patch_training, loss_name):
    model.train(True)
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
# Valid 
@torch.no_grad()
def valid_CNN_Based_Previous(model, criterion, data_loader, device, epoch, save_dir, loss_name):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # print(input_n_20.shape) # (1, 1, 512, 512)
        
        if hasattr(model, 'module'):
            if model.module._get_name() == "Restormer" or model.module._get_name() == "MLPMixer":
                pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(64, 64), sw_batch_size=1, predictor=model.module, overlap=0.5, mode='constant')
            else:
                pred_n_100 = model(input_n_20)

        else :
            if model._get_name() == "Restormer" or model._get_name() == "MLPMixer":
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
# TEST 
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
        if model._get_name() == "Restormer" or model._get_name() == "TED_Net":
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



# GAN Based  ################################################
# 1.WGAN
def train_WGAN_VGG_Previous(model, data_loader, optimizer_G, optimizer_D, device, epoch, patch_training):
    model.Generator.train(True)
    model.Discriminator.train(True)

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
        
        # print("Check = ", input_n_20.max(), input_n_20.min(), input_n_20.dtype, input_n_20.shape)

        # Discriminator, 4 time more training than Generator
        optimizer_D.zero_grad()
        model.Discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(4):
            d_loss, gp_loss = model.d_loss(input_n_20, input_n_100, gp=True, return_gp=True)
            d_loss.backward()
            optimizer_D.step()

        # Generator, perceptual loss
        optimizer_G.zero_grad()
        model.Generator.zero_grad()     # Same as optimizer zero grad()
        g_loss, p_loss = model.g_loss(input_n_20, input_n_100, perceptual=True, return_p=True)
        g_loss.backward()
        optimizer_G.step()
        
        metric_logger.update(g_loss=g_loss, d_loss=d_loss, p_loss=p_loss, gp_loss=gp_loss)
        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_WGAN_VGG_Previous(model, criterion, data_loader, device, epoch, save_dir):
    model.Generator.eval()
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        pred_n_100 = model.Generator(input_n_20)
            
        L1_loss = criterion(pred_n_100, input_n_100)
        loss_value = L1_loss.item()
        metric_logger.update(L1_loss=loss_value)

    # print("Check == ", pred_n_100.shape, pred_n_100.max(), pred_n_100.min())
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
def test_WGAN_VGG_Previous(model, data_loader, device, save_dir):
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


# 2.SACNN 
def train_SACNN_Previous_3D(model, data_loader, optimizer_G, optimizer_D, device, epoch, patch_training):
    model.Generator.train(True)
    model.Discriminator.train(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train: [epoch:{}]'.format(epoch)
    print_freq = 10  

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        if patch_training: 
            input_n_20  = torch.cat([ batch_data[i]['n_20']  for i in range(8) ]).float().permute(0,1,4,2,3).to(device)  # 8 is patch_nums
            input_n_100 = torch.cat([ batch_data[i]['n_100'] for i in range(8) ]).float().permute(0,1,4,2,3)[:,:,1,:,:].to(device)  # (8*batch, C(=1), 64, 64) or (8*batch, C(=1), D(=3), H(=64), W(=64))

        else :
            input_n_20   = batch_data['n_20'].float().permute(0,1,4,2,3).to(device)
            input_n_100  = batch_data['n_100'].float().permute(0,1,4,2,3)[:,:,1,:,:].to(device)


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
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float().permute(0,1,4,2,3)
        input_n_100  = batch_data['n_100'].to(device).float().permute(0,1,4,2,3)
                     

        # zero        = torch.zeros((1,1,1,512,512), device='cuda')
        # paded_input = torch.cat([zero, input_n_20, zero], dim=2)        

        pred_n_100 = sliding_window_inference(inputs=input_n_20, roi_size=(3, 64, 64), sw_batch_size=1, predictor=model.Generator, overlap=0.25, mode='gaussian')

        L1_loss = criterion(pred_n_100, input_n_100)

        loss_value = L1_loss.item()

        metric_logger.update(L1_loss=loss_value)
    
    # np.save('/workspace/sunggu/c.npy', pred_n_100.cpu().detach().numpy()) # 이상없음
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    # Denormalize
    input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20[:, :, 34, :, :])).clip(min=0, max=80)      # select depth 36 brain slice
    input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100[:, :, 34, :, :])).clip(min=0, max=80)
    pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100[:, :, 34, :, :])).clip(min=0, max=80) 

    # PNG Save
    print(save_dir+'epoch_'+str(epoch)+'_input_n_20.png')
    
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_gt_n_100.png',   input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# 수정중
from monai.transforms import SaveImage
@torch.no_grad()
def test_SACNN_Previous_3D(model, data_loader, device, save_dir):
    # switch to evaluation mode
    model.Generator.eval()
    cnt = 0
    # compute PSNR, SSIM, RMSE
    ori_psnr_avg,  ori_ssim_avg,  ori_rmse_avg  = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    gt_psnr_avg,   gt_ssim_avg,   gt_rmse_avg   = 0, 0, 0

    saver = SaveImage(output_dir=save_dir, output_ext=".nii.gz", resample=False, separate_folder=False)

    iterator = tqdm(data_loader, desc='TEST: ', file=sys.stdout, mininterval=50)    
    for batch_data in iterator:
        
        # forward pass
        input_n_20   = batch_data['n_20'].to(device).float().permute(0,1,4,2,3)
        input_n_100  = batch_data['n_100'].to(device).float().permute(0,1,4,2,3)
        
        os.makedirs(save_dir + batch_data['path_n_20'][0].split('/')[7], mode=0o777, exist_ok=True) # png   save folder

        # print("c1 == ",  input_n_20.shape) ([1, 1, 56, 512, 512])
        # print("c2 == ",  input_n_100.shape)

        # # pad_input
        zero      = torch.zeros([1, 1, 1, 512, 512]).to(device)
        pad_input = torch.cat([zero, input_n_20, zero], dim=2)

        # Forward Generator
        empty = torch.zeros_like(input_n_20).to(device)
        for i in range(input_n_20.shape[2]):
            input      = pad_input[:, :, i:i+3, :, :]
            pred_n_100 = sliding_window_inference(inputs=input, roi_size=(3, 64, 64), sw_batch_size=1, predictor=model.Generator, padding_mode='constant', overlap=0.25, mode='gaussian')
            empty[:,:,i,:,:] = pred_n_100[:,:,1,:,:]

        pred_n_100 = empty
        print("LAST pred_n_100 == ", pred_n_100.shape)

        # NII Save [C,H,W,[D]].
        input_low_dcm    = dicom_denormalize(fn_tonumpy3d(input_n_20))[0]
        input_high_dcm   = dicom_denormalize(fn_tonumpy3d(input_n_100))[0]
        pred_n_100_dcm   = dicom_denormalize(fn_tonumpy3d(pred_n_100))[0]      
        
        saver(input_low_dcm,  {"filename_or_obj":save_dir.replace('/png/', '/nii/')+batch_data['path_n_20'][0].split('/')[7]  +'/'+ str(cnt) +'_gt_n_20.nii.gz'})        
        saver(input_high_dcm, {"filename_or_obj":save_dir.replace('/png/', '/nii/')+batch_data['path_n_100'][0].split('/')[7] +'/'+ str(cnt) +'_gt_n_100.nii.gz'})        
        saver(pred_n_100_dcm, {"filename_or_obj":save_dir.replace('/png/', '/nii/')+batch_data['path_n_20'][0].split('/')[7]  +'/'+ str(cnt) +'_pred_n_100.nii.gz'})        
        cnt += 1
        # Metric
        print("!!! == ", input_low_dcm.shape)
        original_result, pred_result, gt_result = compute_measure_3D(x=torch.tensor(input_low_dcm).unsqueeze(0), y=torch.tensor(input_high_dcm).unsqueeze(0), pred=torch.tensor(pred_n_100_dcm).unsqueeze(0), data_range=4095.0)
        ori_psnr_avg  += original_result[0]
        ori_ssim_avg  += original_result[1]
        ori_rmse_avg  += original_result[2]
        pred_psnr_avg += pred_result[0]
        pred_ssim_avg += pred_result[1]
        pred_rmse_avg += pred_result[2]
        gt_psnr_avg   += gt_result[0]
        gt_ssim_avg   += gt_result[1]
        gt_rmse_avg   += gt_result[2]


    print('\n')
    print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(data_loader), ori_ssim_avg/len(data_loader), ori_rmse_avg/len(data_loader)))
    print('\n')
    print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(data_loader), pred_ssim_avg/len(data_loader), pred_rmse_avg/len(data_loader)))        
    print('\n')
    print('GT === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(gt_psnr_avg/len(data_loader), gt_ssim_avg/len(data_loader), gt_rmse_avg/len(data_loader)))        



# 3.DUGAN
def train_DUGAN_Previous(model, data_loader, optimizer_G, optimizer_Img_D, optimizer_Grad_D, device, epoch, patch_training):
    model.Generator.train(True)
    model.Img_Discriminator.train(True)
    model.Grad_Discriminator.train(True)

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
        low_dose  = input_n_20
        full_dose = input_n_100
        gen_full_dose       = model.Generator(low_dose)
        grad_gen_full_dose  = model.sobel(gen_full_dose)
        grad_low_dose       = model.sobel(low_dose)
        grad_full_dose      = model.sobel(full_dose)   

        # Discriminator
        optimizer_Img_D.zero_grad()
        model.Img_Discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(1):
            Img_D_loss, Img_D_dict   = model.train_Img_Discriminator(full_dose, low_dose, gen_full_dose, prefix='Img', n_iter=epoch)
            Img_D_loss.backward()
            optimizer_Img_D.step()
        
        optimizer_Grad_D.zero_grad()
        model.Grad_Discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(1):
            Grad_D_loss, Grad_D_dict = model.train_Grad_Discriminator(grad_full_dose, grad_low_dose, grad_gen_full_dose, prefix='Grad', n_iter=epoch)
            Grad_D_loss.backward()
            optimizer_Grad_D.step()        

        # Generator
        optimizer_G.zero_grad()
        model.Generator.zero_grad()     # Same as optimizer zero grad()
        img_gen_enc, img_gen_dec   = model.Img_Discriminator(gen_full_dose)
        img_gen_loss               = model.gan_metric(img_gen_enc, 1.) + model.gan_metric(img_gen_dec, 1.)

        grad_gen_enc, grad_gen_dec = model.Grad_Discriminator(grad_gen_full_dose)
        grad_gen_loss              = model.gan_metric(grad_gen_enc, 1.) + model.gan_metric(grad_gen_dec, 1.)

        adv_loss  = 0.1*img_gen_loss + 0.1*grad_gen_loss 
        pix_loss  = 1.0*F.mse_loss(gen_full_dose, full_dose) # + F.l1_loss(gen_full_dose, full_dose) 논문에서는 없었음...
        grad_loss = 20.0*F.l1_loss(grad_gen_full_dose, grad_full_dose)

        G_loss = adv_loss + pix_loss + grad_loss
                 
        G_loss.backward()        
        optimizer_G.step()


        G_dict = {}
        G_dict.update({
            'loss/img_gen_loss': img_gen_loss,
            'loss/grad_gen_loss': grad_gen_loss,
            'loss/pix_loss': pix_loss,
            'loss/grad_loss': grad_loss,
        })

        # msg_dict.update({
        #     'enc/grad_gen_enc': grad_gen_enc,
        #     'dec/grad_gen_dec': grad_gen_dec,
        #     'loss/grad_gen_loss': grad_gen_loss,
        #     'enc/img_gen_enc': img_gen_enc,
        #     'dec/img_gen_dec': img_gen_dec,
        #     'loss/img_gen_loss': img_gen_loss,
        #     'loss/pix': pix_loss,
        #     'loss/l1': l1_loss,
        #     'loss/grad': grad_loss,
        # })

        metric_logger.update(**G_dict)
        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_DUGAN_Previous(model, criterion, data_loader, device, epoch, save_dir):
    model.Generator.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # print(input_n_20.shape) # (1, 1, 512, 512)

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


# 4.MAP_NN
def train_MAP_NN_Previous(model, data_loader, optimizer_G, optimizer_D, device, epoch, patch_training):
    model.Generator.train(True)
    model.Discriminator.train(True)

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
        
        # print("Check = ", input_n_20.max(), input_n_20.min(), input_n_20.dtype, input_n_20.shape)

        # Discriminator, 4 time more training than Generator
        optimizer_D.zero_grad()
        model.Discriminator.zero_grad()  # Same as optimizer zero grad()
        for _ in range(4):
            d_loss, gp_loss = model.d_loss(input_n_20, input_n_100, gp=True, return_gp=True)
            d_loss.backward()
            optimizer_D.step()

        # Generator, perceptual loss
        optimizer_G.zero_grad()
        model.Generator.zero_grad()     # Same as optimizer zero grad()
        g_loss, adv_loss, mse_loss, edge_loss = model.g_loss(input_n_20, input_n_100)
        g_loss.backward()
        optimizer_G.step()
        
        metric_logger.update(g_loss=g_loss, d_loss=d_loss, gp_loss=gp_loss, adv_loss=adv_loss, mse_loss=mse_loss, edge_loss=edge_loss)
        metric_logger.update(lr=optimizer_G.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_MAP_NN_Previous(model, criterion, data_loader, device, epoch, save_dir):
    model.Generator.eval()
    model.Discriminator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid: [epoch:{}]'.format(epoch)
    print_freq = 200    

    os.makedirs(save_dir, mode=0o777, exist_ok=True)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        input_n_20   = batch_data['n_20'].to(device).float()
        input_n_100  = batch_data['n_100'].to(device).float()
        
        # print(input_n_20.shape) # (1, 1, 512, 512)

        pred_n_100 = model.Generator(input_n_20)     
            
        L1_loss = criterion(pred_n_100, input_n_100)

        loss_value = L1_loss.item()

        metric_logger.update(L1_loss=loss_value)

    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    # # Denormalize
    # input_n_20   = dicom_denormalize(fn_tonumpy(input_n_20)).clip(min=0, max=80)
    # input_n_100  = dicom_denormalize(fn_tonumpy(input_n_100)).clip(min=0, max=80)
    # pred_n_100   = dicom_denormalize(fn_tonumpy(pred_n_100)).clip(min=0, max=80)     

    # # PNG Save
    # print(save_dir+'epoch_'+str(epoch)+'_input_n_20.png')
    
    # plt.imsave(save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(save_dir+'epoch_'+str(epoch)+'_gt_n_100.png', input_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)
    # plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray", vmin=0, vmax=80)

    # window 수정
    # Denormalize
    input_n_20   = fn_tonumpy(input_n_20).clip(min=0, max=1)
    input_n_100  = fn_tonumpy(input_n_100).clip(min=0, max=1)
    pred_n_100   = fn_tonumpy(pred_n_100).clip(min=0, max=1) 

    # PNG Save
    print(save_dir+'epoch_'+str(epoch)+'_input_n_20.png')
    
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_input_n_20.png', input_n_20.squeeze(), cmap="gray")
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_gt_n_100.png', input_n_100.squeeze(), cmap="gray")
    plt.imsave(save_dir+'epoch_'+str(epoch)+'_pred_n_100.png', pred_n_100.squeeze(), cmap="gray")


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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

# 1. TEST Unet GAN vs Unet
# 2. TEST Restomer vs Unet 
# 3. TEST 해상도 유지 SPADE vs 업샘플 (Transformer_Generator vs Restormer_Decoder/Uformer_Decoder)

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
