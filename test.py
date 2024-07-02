import os
import argparse
import datetime
import time
import json
import random
import torch
import numpy as np
import utils
from dataloaders import get_test_dataloader
from models import get_model
from losses import get_loss
from engine import *



def get_args_parser():
    parser = argparse.ArgumentParser('MTD-GAN Deep-Learning Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',               default="amc", type=str, help='dataset name')
    parser.add_argument('--dataset-type-test',     default="window_patch", type=str, help='dataset type test name')
    
    parser.add_argument('--test-batch-size',       default=72, type=int)
    parser.add_argument('--test-num-workers',      default=10, type=int)

    # Model parameters
    parser.add_argument('--model',                 default='Sequence_SkipHidden_Unet_ALL',  type=str, help='model name')    
    parser.add_argument('--loss',                  default='Sequence_SkipHidden_Unet_loss', type=str, help='loss name')    
    parser.add_argument('--method',                default='', help='multi-task weighting name')

    # DataParrel or Single GPU 
    parser.add_argument('--multi-gpu-mode',        default='DataParallel', choices=['Single', 'DataParallel'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',                default='cuda', help='device to use for training / testing')
    
    # Continue Training
    parser.add_argument('--resume',                default='',  help='resume from checkpoint')  # '' = None    

    # Validation setting
    parser.add_argument('--print-freq',            default=10, type=int, metavar='N', help='print frequency (default: 10)')

    # Prediction and Save setting
    parser.add_argument('--checkpoint-dir',        default='', help='path where to save checkpoint or output')
    parser.add_argument('--save-dir',              default='', help='path where to prediction PNG save')
    parser.add_argument('--epoch',                 default=10, type=int)

    # Memo
    parser.add_argument('--memo',                  default='', help='memo for script')
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
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    print('Available CPUs: ', os.cpu_count())
    utils.print_args_test(args)
    device = torch.device(args.device)

    # Dataloader
    data_loader_test = get_test_dataloader(name=args.dataset, args=args)   

    # Model
    model = get_model(name=args.model)

    # Multi-GPU
    if args.multi_gpu_mode == 'DataParallel':
        if args.model == 'WGAN_VGG' or args.model == 'MAP_NN' or args.model == 'MTD_GAN' or args.model == 'Ablation_CLS' or args.model == 'Ablation_SEG' or args.model == 'Ablation_CLS_SEG' or args.model == 'Ablation_CLS_REC' or args.model == 'Ablation_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC_NDS' or args.model == 'Ablation_CLS_SEG_REC_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC_ResFFT' or args.model == 'MTD_GAN_All_One' or args.model == 'MTD_GAN_Method':
            model.Generator             = torch.nn.DataParallel(model.Generator)         
            model.Discriminator         = torch.nn.DataParallel(model.Discriminator)
            model.Generator.to(device)   
            model.Discriminator.to(device)   
        elif args.model == 'DU_GAN':
            model.Generator             = torch.nn.DataParallel(model.Generator)         
            model.Image_Discriminator   = torch.nn.DataParallel(model.Image_Discriminator)
            model.Grad_Discriminator    = torch.nn.DataParallel(model.Grad_Discriminator)
            model.Generator.to(device)   
            model.Image_Discriminator.to(device)   
            model.Grad_Discriminator.to(device)   
        else :
            model = torch.nn.DataParallel(model)
            model.to(device)            
    else :
        model.to(device)

    # Loss
    loss = get_loss(name=args.loss)

    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint['model_state_dict'] = {k.replace('.module', ''):v for k,v in checkpoint['model_state_dict'].items()} # fix loading multi-gpu 
        model.load_state_dict(checkpoint['model_state_dict'])

    start_time = time.time()

    # CNN based
        # Previous
    if args.model == 'RED_CNN' or args.model == 'ED_CNN': 
        test_stats = test_CNN_Based_Previous(model, loss, data_loader_test, device, args.save_dir)
        print("Averaged test stats: ", test_stats)
    
    # Transformer based
    elif args.model == 'CTformer' or args.model == 'Restormer': 
        test_stats = test_TR_Based_Previous(model, loss, data_loader_test, device, args.save_dir)
        print("Averaged test stats: ", test_stats)

    # DN based
    elif args.model == 'DDPM' or args.model == 'DDIM' or args.model == 'PNDM' or args.model == 'DPM': 
        test_stats = test_DN_Previous(model, loss, data_loader_test, device, args.save_dir)
        print("Averaged test stats: ", test_stats)

    # GAN based
    elif args.model == 'WGAN_VGG': 
        test_stats = test_WGAN_VGG_Previous(model, loss, data_loader_test, device, args.save_dir)
        print("Averaged test stats: ", test_stats)

    elif args.model == 'MAP_NN': 
        test_stats = test_MAP_NN_Previous(model, loss, data_loader_test, device, args.save_dir)
        print("Averaged test stats: ", test_stats)

    elif args.model == 'DU_GAN': 
        test_stats = test_DUGAN_Previous(model, loss, data_loader_test, device, args.save_dir)
        print("Averaged test stats: ", test_stats)

    # Ours
    elif args.model == 'MTD_GAN' or args.model == 'Ablation_CLS' or args.model == 'Ablation_SEG' or args.model == 'Ablation_CLS_SEG' or args.model == 'Ablation_CLS_REC' or args.model == 'Ablation_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC_NDS' or args.model == 'Ablation_CLS_SEG_REC_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC_ResFFT' or args.model == 'MTD_GAN_All_One' or args.model == 'MTD_GAN_Method':
        test_stats = test_MTD_GAN_Ours(model, loss, data_loader_test, device, args.save_dir)
        print("Averaged test stats: ", test_stats)  
                
    # Log & Save
    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': args.epoch}

    with open(args.checkpoint_dir + "/test_log.txt", "a") as f:
        f.write(json.dumps(log_stats) + "\n")

    print('***********************************************')
    print("Finish...!")
    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('TEST time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MTD-GAN evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()        
    
    os.makedirs(args.checkpoint_dir + "/args", exist_ok =True)
    os.makedirs(args.save_dir, mode=0o777, exist_ok=True)

    # Save args to json
    if not os.path.isfile(args.checkpoint_dir + "/args/test_args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json"):
        with open(args.checkpoint_dir + "/args/test_args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    main(args)
