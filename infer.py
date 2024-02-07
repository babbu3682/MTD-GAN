import os
import argparse
import datetime
import time
import json
from pathlib import Path
import random
import torch
import numpy as np

import utils
from create_datasets.prepare_datasets import build_dataset_test
from create_model import create_model
from losses import create_criterion
from engine import *


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('FD-GAN Deep-Learning Train and Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--data-folder-dir', default="/workspace/sunggu/1.Hemorrhage/4.Dose_img2img/datasets/[sinogram]Brain_3mm_DCM", type=str, help='dataset folder dirname')    
    
    # Model parameters
    parser.add_argument('--model-name',      default='Sequence_SkipHidden_Unet_loss', type=str, help='model name')      
    parser.add_argument('--criterion',       default='Sequence_SkipHidden_Unet_loss', type=str, help='criterion name')    

    # Test Option
    parser.add_argument('--windowing',       default="FALSE",   type=str2bool, help='apply windowing')  

    # DataLoader setting
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem',    action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # DataParrel or Single GPU train
    parser.add_argument('--device',               default='cuda', help='device to use for training / testing')
    parser.add_argument('--cuda-device-order',    default='PCI_BUS_ID', type=str, help='cuda_device_order')
    parser.add_argument('--cuda-visible-devices', default='0', type=str, help='cuda_visible_devices')

    # Continue Training
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    # Prediction and Save setting
    parser.add_argument('--png-save-dir',   default='', help='path where to prediction PNG save')

    return parser



# fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)



def main(args):       

    utils.print_args_test(args)
    device = torch.device(args.device)

    print("Loading dataset ....")
    dataset_test, collate_fn_test = build_dataset_test(args=args)
    data_loader_test              = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_fn_test)

    # Select Loss
    print(f"Creating criterion: {args.criterion}")
    criterion = create_criterion(name=args.criterion)

    # Select Model
    print(f"Creating model: {args.model_name}")
    model = create_model(name=args.model_name)
    print(model)


    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint['model_state_dict'] = {k.replace('.module', ''):v for k,v in checkpoint['model_state_dict'].items()} # fix loading multi-gpu 
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load Only Generator(=Denoiser) weights
        # generator = {k: v for k, v in checkpoint['model_state_dict'].items() if "Generator." in k}
        # generator = {k.replace('Generator.', ''):v for k,v in generator.items()} 
        # model.Generator.load_state_dict(generator)   

    # Cuda
    model.to(device)
    start_time = time.time()

    # CNN based
        # Previous
    if args.model_name == 'RED_CNN' or args.model_name == 'ED_CNN': 
        test_stats = test_CNN_Based_Previous(model, criterion, data_loader_test, device, args.png_save_dir)
        print("Averaged test stats: ", test_stats)
    # Transformer based
    elif args.model_name == 'CTformer' or args.model_name == 'Restormer': 
        test_stats = test_Transformer_Based_Previous(model, criterion, data_loader_test, device, args.png_save_dir)
        print("Averaged test stats: ", test_stats)

        # Ours
    elif args.model_name == 'SPADE_UNet' or args.model_name == 'SPADE_UNet_Upgrade_3' or args.model_name == 'ResFFT_LFSPADE': 
        test_stats = test_CNN_Based_Ours(model, criterion, data_loader_test, device, args.png_save_dir)
        print("Averaged test stats: ", test_stats)



    # GAN based
        # Previous
    elif args.model_name == 'WGAN_VGG': 
        test_stats = test_WGAN_VGG_Previous(model, criterion, data_loader_test, device, args.png_save_dir)
        print("Averaged test stats: ", test_stats)

    elif args.model_name == 'MAP_NN': 
        test_stats = test_MAP_NN_Previous(model, criterion, data_loader_test, device, args.png_save_dir)
        print("Averaged test stats: ", test_stats)

    elif args.model_name == 'Markovian_Patch_GAN':         
        test_stats = test_Markovian_Patch_GAN_Previous(model, criterion, data_loader_test, device, args.png_save_dir)
        print("Averaged test stats: ", test_stats)

    elif args.model_name == 'DU_GAN': 
        test_stats = test_DUGAN_Previous(model, criterion, data_loader_test, device, args.png_save_dir)
        print("Averaged test stats: ", test_stats)

        # Ours
    elif args.model_name == 'FDGAN' or args.model_name == 'FDGAN_domain': 
        test_stats = test_FDGAN_Ours(model, criterion, data_loader_test, device, args.png_save_dir)    
        print("Averaged test stats: ", test_stats)
   
    elif args.model_name == 'MTD_GAN' or args.model_name == 'Ablation_A' or args.model_name == 'Ablation_B' or args.model_name == 'Ablation_C' or args.model_name == 'Ablation_D' or args.model_name == 'Ablation_E' or args.model_name == 'Ablation_D2':
        infer_MTD_GAN_Ours(model, criterion, data_loader_test, device, args.png_save_dir)

    else :
        pass
        

    print('***********************************************')
    print("Finish...!")
    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FD-GAN training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()        
    
    os.environ["CUDA_DEVICE_ORDER"]     =  args.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"]  =  args.cuda_visible_devices       

    main(args)


