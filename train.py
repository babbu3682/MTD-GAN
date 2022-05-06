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
from create_datasets.prepare_datasets import build_dataset
from create_model import create_model
from lr_scheduler import create_scheduler
from optimizers import create_optim
from losses import create_criterion
from engine import *


def fix_optimizer(optimizer):
    # Optimizer Error fix...!
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

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
    # parser.add_argument('--data-set', default='CIFAR10', type=str, help='dataset name')    
    parser.add_argument('--data-folder-dir', default="/workspace/sunggu/1.Hemorrhage/4.Dose_img2img/datasets/[sinogram]Brain_3mm_DCM", type=str, help='dataset folder dirname')    

    # Model parameters
    parser.add_argument('--model-name',      default='Sequence_SkipHidden_Unet_ALL',  type=str, help='model name')    
    parser.add_argument('--criterion',       default='Sequence_SkipHidden_Unet_loss', type=str, help='criterion name')    
    # parser.add_argument('--criterion_mode',  default='none', type=str,  help='criterion mode')  

    # Training Option
    parser.add_argument('--patch_training',  default="FALSE",   type=str2bool, help='patch_training')    
    parser.add_argument('--multiple_GT',     default="FALSE",   type=str2bool, help='multiple ground truth')  
    parser.add_argument('--windowing',       default="FALSE",   type=str2bool, help='apply windowing')  

    # DataLoader setting
    parser.add_argument('--batch-size',  default=72, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem',    action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Optimizer parameters
    parser.add_argument('--optimizer', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "AdamW"')
    
    # Learning rate and schedule and Epoch parameters
    parser.add_argument('--lr-scheduler', default='poly_lr', type=str, metavar='lr_scheduler', help='lr_scheduler (default: "poly_learning_rate"')
    parser.add_argument('--epochs', default=1000, type=int, help='Upstream 1000 epochs, Downstream 500 epochs')  
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode',       default='DataParallel', choices=['DataParallel', 'Single'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',               default='cuda', help='device to use for training / testing')
    parser.add_argument('--cuda-device-order',    default='PCI_BUS_ID', type=str, help='cuda_device_order')
    parser.add_argument('--cuda-visible-devices', default='0', type=str, help='cuda_visible_devices')

    # Continue Training
    parser.add_argument('--resume',           default='',  help='resume from checkpoint')  # '' = None
    parser.add_argument('--from-pretrained',  default='',  help='pre-trained from checkpoint')
    
    # Validation setting
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save-checkpoint-every', default=2, type=int, help='save the checkpoints every n epochs')  

    # Prediction and Save setting
    parser.add_argument('--checkpoint-dir', default='', help='path where to save checkpoint or output')
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

    utils.print_args(args)
    device = torch.device(args.device)

    print("Loading dataset ....")
    dataset_train, collate_fn_train = build_dataset(training_mode='train',  args=args)   
    dataset_valid, collate_fn_valid = build_dataset(training_mode='valid',  args=args)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,   pin_memory=args.pin_mem, drop_last=True,  collate_fn=collate_fn_train)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1,               num_workers=args.num_workers, shuffle=True,   pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_fn_valid) 

    # Select Loss
    print(f"Creating criterion: {args.criterion}")
    criterion = create_criterion(name=args.criterion)

    # Select Model
    print(f"Creating model: {args.model_name}")
    print(f"Pretrained model: {args.from_pretrained}")
    model = create_model(name=args.model_name)
    print(model)


    # Optimizer & LR Schedule
    if args.model_name == 'WGAN_VGG' or args.model_name == 'MAP_NN':
        optimizer_G    = create_optim(name=args.optimizer, model=model.Generator, args=args)
        optimizer_D    = create_optim(name=args.optimizer, model=model.Discriminator, args=args)
        lr_scheduler_G = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_G, args=args)
        lr_scheduler_D = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_D, args=args)
    elif args.model_name == 'Markovian_Patch_GAN':
        optimizer_G    = create_optim(name=args.optimizer, model=model.Generator, args=args); args.lr = args.lr * 4; 
        optimizer_D    = create_optim(name=args.optimizer, model=model.Discriminator, args=args)
        lr_scheduler_G = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_G, args=args)
        lr_scheduler_D = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_D, args=args)        
    elif args.model_name == 'DU_GAN':
        optimizer_G         = create_optim(name=args.optimizer,model=model.Generator, args=args)
        optimizer_Img_D     = create_optim(name=args.optimizer,model=model.Image_Discriminator, args=args)
        optimizer_Grad_D    = create_optim(name=args.optimizer,model=model.Grad_Discriminator, args=args)
        lr_scheduler_G      = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_G, args=args)
        lr_scheduler_Img_D  = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_Img_D, args=args)
        lr_scheduler_Grad_D = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_Grad_D, args=args)
    elif args.model_name == 'FSGAN':
        optimizer_G         = create_optim(name=args.optimizer,model=model.Generator, args=args)
        optimizer_Low_D     = create_optim(name=args.optimizer,model=model.Low_discriminator, args=args)
        optimizer_High_D    = create_optim(name=args.optimizer,model=model.High_discriminator, args=args)
        lr_scheduler_G      = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_G, args=args)
        lr_scheduler_Low_D  = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_Low_D, args=args)
        lr_scheduler_High_D = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_High_D, args=args)
    elif args.model_name == 'FDGAN_PatchGAN':
        optimizer_G             = create_optim(name=args.optimizer,model=model.Generator, args=args)
        optimizer_Image_D       = create_optim(name=args.optimizer,model=model.Image_discriminator, args=args)
        optimizer_Fourier_D     = create_optim(name=args.optimizer,model=model.Fourier_discriminator, args=args)
        lr_scheduler_G          = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_G, args=args)
        lr_scheduler_Image_D    = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_Image_D, args=args)
        lr_scheduler_Fourier_D  = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_Fourier_D, args=args)
    elif args.model_name == 'FDGAN' or args.model_name == "FDGAN_domain":
        optimizer_G             = create_optim(name=args.optimizer,model=model.Generator, args=args)
        optimizer_Image_D       = create_optim(name=args.optimizer,model=model.Image_Discriminator, args=args)
        optimizer_Fourier_D     = create_optim(name=args.optimizer,model=model.Fourier_Discriminator, args=args)
        lr_scheduler_G          = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_G, args=args)
        lr_scheduler_Image_D    = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_Image_D, args=args)
        lr_scheduler_Fourier_D  = create_scheduler(name=args.lr_scheduler, optimizer=optimizer_Fourier_D, args=args)        
    else : 
        optimizer    = create_optim(name=args.optimizer,model=model, args=args)
        lr_scheduler = create_scheduler(name=args.lr_scheduler, optimizer=optimizer, args=args)


    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint['model_state_dict'] = {k.replace('.module', ''):v for k,v in checkpoint['model_state_dict'].items()} # fix loading multi-gpu 
        model.load_state_dict(checkpoint['model_state_dict'])   
        args.start_epoch = checkpoint['epoch'] + 1      
        if args.model_name == 'WGAN_VGG' or args.model_name == 'MAP_NN' or args.model_name == 'Markovian_Patch_GAN':
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])    
            lr_scheduler_D.load_state_dict(checkpoint['lr_scheduler_D'])      
            # Optimizer Error fix...!
            fix_optimizer(optimizer_G)
            fix_optimizer(optimizer_D)
        elif args.model_name == 'DU_GAN':
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_Img_D.load_state_dict(checkpoint['optimizer_Img_D'])
            optimizer_Grad_D.load_state_dict(checkpoint['optimizer_Grad_D'])
            lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])    
            lr_scheduler_Img_D.load_state_dict(checkpoint['lr_scheduler_Img_D'])        
            lr_scheduler_Grad_D.load_state_dict(checkpoint['lr_scheduler_Grad_D'])      
            # Optimizer Error fix...!
            fix_optimizer(optimizer_G)
            fix_optimizer(optimizer_Img_D)
            fix_optimizer(optimizer_Grad_D)
        elif args.model_name == 'FSGAN':       
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_Low_D.load_state_dict(checkpoint['optimizer_Low_D'])
            optimizer_High_D.load_state_dict(checkpoint['optimizer_High_D'])
            lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
            lr_scheduler_Low_D.load_state_dict(checkpoint['lr_scheduler_Low_D'])
            lr_scheduler_High_D.load_state_dict(checkpoint['lr_scheduler_High_D'])
            # Optimizer Error fix...!
            fix_optimizer(optimizer_G)
            fix_optimizer(optimizer_Low_D)
            fix_optimizer(optimizer_High_D)
        elif args.model_name == 'FDGAN_PatchGAN' or args.model_name == 'FDGAN' or args.model_name == "FDGAN_domain":              
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_Image_D.load_state_dict(checkpoint['optimizer_Image_D'])
            optimizer_Fourier_D.load_state_dict(checkpoint['optimizer_Fourier_D'])
            lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
            lr_scheduler_Image_D.load_state_dict(checkpoint['lr_scheduler_Image_D'])
            lr_scheduler_Fourier_D.load_state_dict(checkpoint['lr_scheduler_Fourier_D'])
            # Optimizer Error fix...!
            fix_optimizer(optimizer_G)
            fix_optimizer(optimizer_Image_D)
            fix_optimizer(optimizer_Fourier_D)       
        else :
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # Optimizer Error fix...!
            fix_optimizer(optimizer)


    # Multi-GPU
    if args.multi_gpu_mode == 'DataParallel':
        if args.model_name == 'WGAN_VGG' or args.model_name == 'MAP_NN' or args.model_name == 'Markovian_Patch_GAN':
            model.Generator             = torch.nn.DataParallel(model.Generator)         
            model.Discriminator         = torch.nn.DataParallel(model.Discriminator)
            model.Generator.to(device)   
            model.Discriminator.to(device)   
        elif args.model_name == 'DU_GAN':
            model.Generator             = torch.nn.DataParallel(model.Generator)         
            model.Image_Discriminator   = torch.nn.DataParallel(model.Image_Discriminator)
            model.Grad_Discriminator    = torch.nn.DataParallel(model.Grad_Discriminator)
            model.Generator.to(device)   
            model.Image_Discriminator.to(device)   
            model.Grad_Discriminator.to(device)   
        elif args.model_name == 'FDGAN_PatchGAN':
            model.Generator             = torch.nn.DataParallel(model.Generator)         
            model.Image_discriminator   = torch.nn.DataParallel(model.Image_discriminator)
            model.Fourier_discriminator = torch.nn.DataParallel(model.Fourier_discriminator)
            model.Generator.to(device)   
            model.Image_discriminator.to(device)   
            model.Fourier_discriminator.to(device)  
        elif args.model_name == 'FDGAN' or args.model_name == "FDGAN_domain":
            model.Generator             = torch.nn.DataParallel(model.Generator)         
            model.Image_Discriminator   = torch.nn.DataParallel(model.Image_Discriminator)
            model.Fourier_Discriminator = torch.nn.DataParallel(model.Fourier_Discriminator)
            model.Generator.to(device)   
            model.Image_Discriminator.to(device)   
            model.Fourier_Discriminator.to(device)              
        else :
            model = torch.nn.DataParallel(model)
            model.to(device)            
    else :
        model.to(device)


    # Etc traing setting
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # Whole LOOP Train & Valid #####
    for epoch in range(args.start_epoch, args.epochs):

        # CNN based
            # Previous
        if args.model_name == 'RED_CNN' or args.model_name == 'ED_CNN': 
            train_stats = train_CNN_Based_Previous(model, data_loader_train, optimizer, device, epoch, args.patch_training, args.print_freq, args.batch_size)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_CNN_Based_Previous(model, criterion, data_loader_valid, device, epoch, args.png_save_dir, args.print_freq, args.batch_size)
            print("Averaged valid_stats: ", valid_stats)
        # Transformer based
        elif args.model_name == 'Restormer' or args.model_name == 'CTformer': 
            train_stats = train_Transformer_Based_Previous(model, data_loader_train, optimizer, device, epoch, args.patch_training, args.print_freq, args.batch_size)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_Transformer_Based_Previous(model, criterion, data_loader_valid, device, epoch, args.png_save_dir, args.print_freq, args.batch_size)
            print("Averaged valid_stats: ", valid_stats)


            # Ours
        elif args.model_name == 'ResFFT_LFSPADE' or args.model_name == 'ResFFT_Freq_SPADE_Att' or args.model_name == 'ResFFT_Freq_SPADE_Att_window':
            train_stats = train_CNN_Based_Ours(model, criterion, data_loader_train, optimizer, device, epoch, args.patch_training, args.criterion)
            valid_stats = valid_CNN_Based_Ours(model, criterion, data_loader_valid, device, epoch, args.png_save_dir, args.criterion)


        # GAN based
            # Previous        
        elif args.model_name == 'WGAN_VGG': 
            train_stats = train_WGAN_VGG_Previous(model, data_loader_train, optimizer_G, optimizer_D, device, epoch, args.patch_training, args.print_freq, args.batch_size)            
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_WGAN_VGG_Previous(model, criterion, data_loader_valid, device, epoch, args.png_save_dir, args.print_freq, args.batch_size)
            print("Averaged valid_stats: ", valid_stats)

        elif args.model_name == 'MAP_NN': 
            train_stats = train_MAP_NN_Previous(model, data_loader_train, optimizer_G, optimizer_D, device, epoch, args.patch_training, args.print_freq, args.batch_size)            
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_MAP_NN_Previous(model, criterion, data_loader_valid, device, epoch, args.png_save_dir, args.print_freq, args.batch_size)
            print("Averaged valid_stats: ", valid_stats)

        elif args.model_name == 'Markovian_Patch_GAN': 
            train_stats = train_Markovian_Patch_GAN_Previous(model, data_loader_train, optimizer_G, optimizer_D, device, epoch, args.patch_training, args.print_freq, args.batch_size)            
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_Markovian_Patch_GAN_Previous(model, criterion, data_loader_valid, device, epoch, args.png_save_dir, args.print_freq, args.batch_size)
            print("Averaged valid_stats: ", valid_stats)

        elif args.model_name == 'DU_GAN': 
            train_stats = train_DUGAN_Previous(model, data_loader_train, optimizer_G, optimizer_Img_D, optimizer_Grad_D, device, epoch, args.patch_training, args.print_freq, args.batch_size)            
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_DUGAN_Previous(model, criterion, data_loader_valid, device, epoch, args.png_save_dir, args.print_freq, args.batch_size)
            print("Averaged valid_stats: ", valid_stats)

            # Ours
        elif args.model_name == "FSGAN": 
            train_stats = train_FSGAN_Previous(model, data_loader_train, optimizer_G, optimizer_Low_D, optimizer_High_D, device, epoch, args.patch_training)            
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_FSGAN_Previous(model, criterion, data_loader_valid, device, epoch, args.png_save_dir)
            print("Averaged valid_stats: ", valid_stats)

        elif args.model_name == "FDGAN_PatchGAN": 
            train_stats = train_FDGAN_PatchGAN_Ours(model, data_loader_train, optimizer_G, optimizer_Image_D, optimizer_Fourier_D, device, epoch, args.patch_training)            
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_FDGAN_PatchGAN_Ours(model, criterion, data_loader_valid, device, epoch, args.png_save_dir)
            print("Averaged valid_stats: ", valid_stats)

        elif args.model_name == "FDGAN" or args.model_name == "FDGAN_domain":
            train_stats = train_FDGAN_Ours(model, data_loader_train, optimizer_G, optimizer_Image_D, optimizer_Fourier_D, device, epoch, args.patch_training, args.print_freq, args.batch_size)            
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_FDGAN_Ours(model, criterion, data_loader_valid, device, epoch, args.png_save_dir, args.print_freq, args.batch_size)
            print("Averaged valid_stats: ", valid_stats)

        else :
            pass
            


        # Save & Prediction png
        if epoch % args.save_checkpoint_every == 0:
            checkpoint_path = args.checkpoint_dir + '/epoch_' + str(epoch) + '_checkpoint.pth'

            if args.model_name == 'WGAN_VGG' or args.model_name == 'MAP_NN' or args.model_name == 'Markovian_Patch_GAN':
                torch.save({
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                    'optimizer_G': optimizer_G.state_dict(), 
                    'optimizer_D': optimizer_D.state_dict(), 
                    'lr_scheduler_G': lr_scheduler_G.state_dict(),
                    'lr_scheduler_D': lr_scheduler_D.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)               

            elif args.model_name == 'DU_GAN':
                torch.save({
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
                torch.save({
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

            elif args.model_name == 'FDGAN_PatchGAN' or args.model_name == 'FDGAN' or args.model_name == "FDGAN_domain":
                torch.save({
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                    'optimizer_G': optimizer_G.state_dict(), 
                    'optimizer_Image_D': optimizer_Image_D.state_dict(), 
                    'optimizer_Fourier_D': optimizer_Fourier_D.state_dict(), 
                    'lr_scheduler_G': lr_scheduler_G.state_dict(),
                    'lr_scheduler_Image_D': lr_scheduler_Image_D.state_dict(),
                    'lr_scheduler_Fourier_D': lr_scheduler_Fourier_D.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)   

            else :
                torch.save({
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                    'optimizer': optimizer.state_dict(), 
                    'lr_scheduler': lr_scheduler.state_dict(), 
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)                                        

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'valid_{k}': v for k, v in valid_stats.items()},
                     'epoch': epoch}

        if args.checkpoint_dir:
            with open(args.checkpoint_dir + "/log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.model_name == 'WGAN_VGG' or args.model_name == 'MAP_NN' or args.model_name == 'Markovian_Patch_GAN':
            lr_scheduler_G.step(epoch)
            lr_scheduler_D.step(epoch)
        elif args.model_name == 'DU_GAN':            
            lr_scheduler_G.step(epoch)
            lr_scheduler_Img_D.step(epoch)
            lr_scheduler_Grad_D.step(epoch)
        elif args.model_name == 'FSGAN':            
            lr_scheduler_G.step(epoch)
            lr_scheduler_Low_D.step(epoch)
            lr_scheduler_High_D.step(epoch)            
        elif args.model_name == 'FDGAN_PatchGAN' or args.model_name == 'FDGAN' or args.model_name == "FDGAN_domain":            
            lr_scheduler_G.step(epoch)
            lr_scheduler_Image_D.step(epoch)
            lr_scheduler_Fourier_D.step(epoch)                                    
        else:    
            lr_scheduler.step(epoch)


    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FD-GAN training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
    os.environ["CUDA_DEVICE_ORDER"]     =  args.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"]  =  args.cuda_visible_devices        
    
    main(args)
