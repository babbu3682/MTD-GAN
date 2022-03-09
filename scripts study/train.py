import argparse
import datetime
import time
import json
import random
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath('/workspace/sunggu'))
sys.path.append(os.path.abspath('/workspace/sunggu/4.Dose_img2img/LowDose_HighDose_Code_Factory'))
sys.path.append(os.path.abspath('/workspace/sunggu/4.Dose_img2img/LowDose_HighDose_Code_Factory/utils'))
sys.path.append(os.path.abspath('/workspace/sunggu/4.Dose_img2img/LowDose_HighDose_Code_Factory/module'))

import torch
import numpy as np
import utils
from datasets.prepare_datasets import build_dataset
from engine import *
from losses import create_criterion
from lr_scheduler import create_scheduler
from create_model import create_model

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Sunggu Deeplearning Train and Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--data-set', default='CIFAR10', type=str, help='dataset name')    

    # DataLoader setting
    parser.add_argument('--batch-size',  default=72, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem',    action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # Model parameters
    parser.add_argument('--model-name',      default='Sequence_SkipHidden_Unet_ALL',  type=str, help='model name')    
    parser.add_argument('--criterion',       default='Sequence_SkipHidden_Unet_loss', type=str, help='criterion name')    
    parser.add_argument('--criterion_mode',  default='none', type=str,  help='criterion mode')
    parser.add_argument('--patch_training',  default="FALSE",   type=str2bool, help='patch_training')    
    parser.add_argument('--multiple_GT',     default="FALSE",   type=str2bool, help='multiple ground truth')    

    # Optimizer parameters
    parser.add_argument('--optimizer', default='AdamW', type=str, metavar='OPTIMIZER', help='Optimizer (default: "AdamW"')
    
    # Learning rate and schedule and Epoch parameters
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_scheduler', default='cosine_annealing_warm_restart', type=str, metavar='lr_scheduler', help='lr_scheduler (default: "cosine_annealing_warm_restart"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')

    # Distributed or DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode', default='DataParallel', choices=['DataParallel', 'DistributedDataParallel', 'Single'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # Continue Training
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')    
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--from_pretrained', default="FALSE",   type=str2bool, help='just start from resume')    
    
    # Validation setting
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--validate-every', default=2, type=int, help='validate and save the checkpoints every n epochs')  

    # Prediction and Save setting
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--save_dir',   default='', help='path where to save, empty for no saving')

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

    utils.print_args(args)
    device = torch.device(args.device)

    print("Loading dataset ....")
    dataset_train, collate_fn_train = build_dataset(training_mode='train',  args=args)   
    dataset_valid, collate_fn_valid = build_dataset(training_mode='valid',  args=args)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,  pin_memory=args.pin_mem, drop_last=True,  collate_fn=collate_fn_train)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1,               num_workers=args.num_workers, shuffle=True,  pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_fn_valid) 

    #### Select Loss
    print(f"Creating criterion: {args.criterion}")
    criterion = create_criterion(name=args.criterion, mode=args.criterion_mode)

    #### Select Model
    print(f"Creating model: {args.model_name}")
    model = create_model(name=args.model_name)
    print(model)

    #### Multi-GPU
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)
        model.to(device)
    else :
        model.to(device)

    #### Optimizer & LR Schedule
    if args.model_name == 'WGAN_VGG' or args.model_name == 'MAP_NN' or args.model_name == 'SACNN':
        optimizer_G    = torch.optim.AdamW(params=model.Generator.parameters(), lr=args.lr)
        optimizer_D    = torch.optim.AdamW(params=model.Discriminator.parameters(), lr=args.lr)                        
        lr_scheduler_G = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_G, args=args)
        lr_scheduler_D = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_D, args=args)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            if not args.from_pretrained:
                optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                optimizer_D.load_state_dict(checkpoint['optimizer_D'])
                lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])    
                lr_scheduler_D.load_state_dict(checkpoint['lr_scheduler_D'])        

    elif args.model_name == 'DU_GAN':
        optimizer_G         = torch.optim.AdamW(params=model.Generator.parameters(), lr=args.lr)
        optimizer_Img_D     = torch.optim.AdamW(params=model.Img_Discriminator.parameters(), lr=args.lr)
        optimizer_Grad_D    = torch.optim.AdamW(params=model.Grad_Discriminator.parameters(), lr=args.lr)
        lr_scheduler_G      = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_G, args=args)
        lr_scheduler_Img_D  = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_Img_D, args=args)
        lr_scheduler_Grad_D = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_Grad_D, args=args)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            if not args.from_pretrained:
                optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                optimizer_Img_D.load_state_dict(checkpoint['optimizer_Img_D'])
                optimizer_Grad_D.load_state_dict(checkpoint['optimizer_Grad_D'])
                lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])    
                lr_scheduler_Img_D.load_state_dict(checkpoint['lr_scheduler_Img_D'])        
                lr_scheduler_Grad_D.load_state_dict(checkpoint['lr_scheduler_Grad_D'])        

    elif args.model_name == 'FSGAN':
        optimizer_G         = torch.optim.AdamW(params=model.Generator.parameters(), lr=args.lr)
        optimizer_Low_D     = torch.optim.AdamW(params=model.Low_discriminator.parameters(), lr=args.lr)
        optimizer_High_D    = torch.optim.AdamW(params=model.High_discriminator.parameters(), lr=args.lr)
        lr_scheduler_G      = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_G, args=args)
        lr_scheduler_Low_D  = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_Low_D, args=args)
        lr_scheduler_High_D = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_High_D, args=args)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            if not args.from_pretrained:
                optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                optimizer_Low_D.load_state_dict(checkpoint['optimizer_Low_D'])
                optimizer_High_D.load_state_dict(checkpoint['optimizer_High_D'])
                lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
                lr_scheduler_Low_D.load_state_dict(checkpoint['lr_scheduler_Low_D'])
                lr_scheduler_High_D.load_state_dict(checkpoint['lr_scheduler_High_D'])

    else : 
        optimizer    = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
        lr_scheduler = create_scheduler(name=args.lr_scheduler, optimizer=optimizer, args=args)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            if not args.from_pretrained:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    #### Resume
    if args.resume:
        print("Loading pre-trained Weight...!")
        model.load_state_dict(checkpoint['model_state_dict'])
        if not args.from_pretrained:
            args.start_epoch = checkpoint['epoch'] + 1
            if 'best_metric' in checkpoint:
                print("Epoch: ", checkpoint['epoch'], " Best Metric ==> ", checkpoint['best_metric'])

    # #### Multi-GPU
    # if args.multi_gpu_mode == 'DataParallel':
    #     model = torch.nn.DataParallel(model)
    #     model.to(device)
    # else :
    #     model.to(device)

    #### Etc traing setting
    output_dir = Path(args.output_dir)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_metric = best_metric1 = best_metric2 = 1000000000.0

    #### Whole LOOP Train & Valid #####
    for epoch in range(args.start_epoch, args.epochs):

        # CNN based
            # Previous
        if args.model_name == 'RED_CNN' or args.model_name == 'ED_CNN' or args.model_name == 'TED_Net' or args.model_name == 'Restormer': 
            train_stats = train_CNN_Based_Previous(model, criterion, data_loader_train, optimizer, device, epoch, args.patch_training, args.criterion)
            valid_stats = valid_CNN_Based_Previous(model, criterion, data_loader_valid, device, epoch, args.save_dir, args.criterion)

            # Ours
        elif args.model_name == 'SPADE_UNet' or args.model_name == 'SPADE_UNet_Upgrade': 
            train_stats = train_CNN_Based_Ours(model, criterion, data_loader_train, optimizer, device, epoch, args.patch_training, args.criterion)
            valid_stats = valid_CNN_Based_Ours(model, criterion, data_loader_valid, device, epoch, args.save_dir, args.criterion)

        # elif args.model_name == 'UNet_Progressive': 
        #     train_stats = train_CNN_Based_Ours_Progress(model, criterion, data_loader_train, optimizer, device, epoch, args.patch_training)
        #     valid_stats = valid_CNN_Based_Ours_Progress(model, criterion, data_loader_valid, device, epoch, args.save_dir)


        # GAN based
            # Previous        
        elif args.model_name == 'WGAN_VGG': 
            train_stats = train_WGAN_VGG_Previous(model, data_loader_train, optimizer_G, optimizer_D, device, epoch, args.patch_training)            
            valid_stats = valid_WGAN_VGG_Previous(model, criterion, data_loader_valid, device, epoch, args.save_dir)

        elif args.model_name == 'MAP_NN': 
            train_stats = train_MAP_NN_Previous(model, data_loader_train, optimizer_G, optimizer_D, device, epoch, args.patch_training)            
            valid_stats = valid_MAP_NN_Previous(model, criterion, data_loader_valid, device, epoch, args.save_dir)

        elif args.model_name == 'SACNN': 
            train_stats = train_SACNN_Previous_3D(model, data_loader_train, optimizer_G, optimizer_D, device, epoch, args.patch_training)            
            valid_stats = valid_SACNN_Previous_3D(model, criterion, data_loader_valid, device, epoch, args.save_dir)

        elif args.model_name == 'DU_GAN': 
            train_stats = train_DUGAN_Previous(model, data_loader_train, optimizer_G, optimizer_Img_D, optimizer_Grad_D, device, epoch, args.patch_training)            
            valid_stats = valid_DUGAN_Previous(model, criterion, data_loader_valid, device, epoch, args.save_dir)

            # Ours
        elif args.model_name == "FSGAN": 
            train_stats = train_FSGAN_Previous(model, data_loader_train, optimizer_G, optimizer_Low_D, optimizer_High_D, device, epoch, args.patch_training)            
            valid_stats = valid_FSGAN_Previous(model, criterion, data_loader_valid, device, epoch, args.save_dir)

        # ETC
        elif args.model_name == 'SACNN_AutoEncoder': 
            train_stats = train_SACNN_AE_Previous_3D(model, criterion, data_loader_train, optimizer, device, epoch, args.patch_training)            
            valid_stats = valid_SACNN_AE_Previous_3D(model, criterion, data_loader_valid, device, epoch, args.save_dir)        

        else :
            pass
            

        ##### Summary #####
        if args.model_name != 'WGAN_VGG' and args.model_name != 'MAP_NN' and args.model_name != 'SACNN' and args.model_name != 'DU_GAN' and args.model_name != 'FSGAN':
            print(f"loss of the network on the {len(dataset_valid)} valid images: {valid_stats['loss']:.3f}%")                
            if valid_stats["loss"] < best_metric1 :    
                best_metric1 = valid_stats["loss"]
                best_metric = best_metric1
                best_metric_epoch = epoch         
            print(f'Min loss: {best_metric:.3f}')    
            print(f'Best Epoch: {best_metric_epoch:.3f}')  

        # Save & Prediction png
        if epoch % args.validate_every == 0:
            save_name = 'epoch_' + str(epoch) + '_checkpoint.pth'
            checkpoint_paths = [output_dir / str(save_name)]
            for checkpoint_path in checkpoint_paths:

                if args.model_name == 'WGAN_VGG' or args.model_name == 'MAP_NN' or args.model_name == 'SACNN':
                    utils.save_on_master({
                        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                        'optimizer_G': optimizer_G.state_dict(), 
                        'optimizer_D': optimizer_D.state_dict(), 
                        'lr_scheduler_G': lr_scheduler_G.state_dict(),
                        'lr_scheduler_D': lr_scheduler_D.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)               

                elif args.model_name == 'DU_GAN':
                    utils.save_on_master({
                        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                        'optimizer_G': optimizer_G.state_dict(), 
                        'optimizer_Img_D': optimizer_Img_D.state_dict(), 
                        'optimizer_Grad_D': optimizer_Grad_D.state_dict(), 
                        'lr_scheduler_G': lr_scheduler_G.state_dict(),
                        'lr_scheduler_Img_D': lr_scheduler_Img_D.state_dict(),
                        'lr_scheduler_Grad_D': lr_scheduler_Grad_D.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)   

                elif args.model_name == 'FSGAN':
                    utils.save_on_master({
                        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                        'optimizer_G': optimizer_G.state_dict(), 
                        'optimizer_Low_D': optimizer_Low_D.state_dict(), 
                        'optimizer_High_D': optimizer_High_D.state_dict(), 
                        'lr_scheduler_G': lr_scheduler_G.state_dict(),
                        'lr_scheduler_Low_D': lr_scheduler_Low_D.state_dict(),
                        'lr_scheduler_High_D': lr_scheduler_High_D.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)   

                else :
                    utils.save_on_master({
                        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                        'optimizer': optimizer.state_dict(), 
                        'lr_scheduler': lr_scheduler.state_dict(), 
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)                                        

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'valid_{k}': v for k, v in valid_stats.items()},
                     'epoch': epoch}

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.model_name == 'MAP_NN' or args.model_name == 'SACNN':
            lr_scheduler_G.step(epoch)
            lr_scheduler_D.step(epoch)
        elif args.model_name == 'WGAN_VGG':            
            pass
        elif args.model_name == 'DU_GAN':            
            lr_scheduler_G.step(epoch)
            lr_scheduler_Img_D.step(epoch)
            lr_scheduler_Grad_D.step(epoch)
        elif args.model_name == 'FSGAN':            
            lr_scheduler_G.step(epoch)
            lr_scheduler_Low_D.step(epoch)
            lr_scheduler_High_D.step(epoch)            
        else:    
            lr_scheduler.step(epoch)


    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sunggu training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)
