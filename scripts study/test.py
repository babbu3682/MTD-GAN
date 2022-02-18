import os
import sys
sys.path.append(os.path.abspath('/workspace/sunggu'))
sys.path.append(os.path.abspath('/workspace/sunggu/4.Dose_img2img'))
sys.path.append(os.path.abspath('/workspace/sunggu/4.Dose_img2img/utils'))
sys.path.append(os.path.abspath('/workspace/sunggu/4.Dose_img2img/module'))

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import random

from pathlib import Path
from create_model import create_model
from datasets.prepare_datasets import build_dataset
from engine import *
from losses import *
import functools

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Sunggu Deeplearning Train and Test script', add_help=False)

    # Model_name
    parser.add_argument('--model-name', default='Sequence_SkipHidden_Unet_loss', type=str, help='model name')      
    parser.add_argument('--training-mode',  default='sinogram', type=str,  help='select sinogram or follow dataset')
    parser.add_argument('--teacher_forcing', default="FALSE",   type=str2bool, help='teacher_forcing')    
    
    # Dataset parameters
    parser.add_argument('--data-set', default='TEST_Sinogram_DCM', type=str, help='dataset name')    
    parser.add_argument('--range-minus1-plus1', type=str2bool, default="FALSE", help='range_minus1_plus1')

    # Continue Training
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    # Validation setting
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Prediction and Save setting
    parser.add_argument('--save_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)
    
    # GPU setting 
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    # parser.add_argument('--gpus', default='0', type=str, help='Gpu index')  

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
    
    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')

    print('***********************************************')
    print('Dataset Name: ', args.data_set)
    print('---------- Model ----------')
    print('Test Epoch: ', checkpoint['epoch'])
    print('Resume From: ', args.resume)

    save_dir = args.save_dir.replace('/low2high/', '_epoch_' + str(checkpoint['epoch']) + '/low2high/')
    print('Output To: ', save_dir)

    print("Loading dataset ....")
    dataset_test, collate_fn_test = build_dataset(training_mode=args.training_mode, args=args)   
    data_loader                   = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_fn_test) #collate_fn_valid

    print(f"Creating model: {args.model_name}")
    model = create_model(name=args.model_name, args=args)
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device(args.device)
    model.to(device)


    # CNN based
        # Previous
    if args.model_name == 'RED_CNN' or args.model_name == 'ED_CNN' or args.model_name == 'TED_Net' or args.model_name == 'Restormer': 
        test_CNN_Based_Previous(model, data_loader, device, args.save_dir)
        
        # Ours
    # elif args.model_name == '': 
        # test_CNN_Based_Previous(model, data_loader, device, args.save_dir)

    # GAN based
        # Previous
    elif args.model_name == 'WGAN_VGG': 
        test_WGAN_VGG_Previous(model, data_loader, device, args.save_dir)

    elif args.model_name == 'MAP_NN': 
        test_MAP_NN_Previous(model, data_loader, device, args.save_dir)

    elif args.model_name == 'SACNN': 
        test_SACNN_Previous_3D(model, data_loader, device, args.save_dir)

    elif args.model_name == 'DU_GAN': 
        test_DUGAN_Previous(model, data_loader, device, args.save_dir)

        # Ours
    elif args.model_name == 'FSGAN':         
        test_FSGAN_Previous(model, data_loader, device, args.save_dir)

    # ETC
    elif args.model_name == 'SACNN_AutoEncoder': 
        test_SACNN_AE_Previous_3D(model, data_loader, device, args.save_dir)

    else :
        pass
        

    print('***********************************************')
    print("Finish...!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sunggu training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()        
    
    main(args)


