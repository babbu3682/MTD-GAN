import os
import argparse
import datetime
import time
import json
import random
import torch
import numpy as np
from collections import defaultdict
import utils
from dataloaders import get_train_dataloader
from models import get_model
from schedulers import get_scheduler
from optimizers import get_optimizer
from losses import get_loss
from engine import *
from module.weight_methods import WeightMethods



def get_args_parser():
    parser = argparse.ArgumentParser('MTD-GAN Deep-Learning Train script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',               default="amc", type=str, help='dataset name')
    parser.add_argument('--dataset-type-train',    default="window_patch", type=str, help='dataset type train name')
    parser.add_argument('--dataset-type-valid',    default="window_patch", type=str, help='dataset type valid name')
    parser.add_argument('--batch-size',            default=72, type=int)
    parser.add_argument('--train-num-workers',     default=10, type=int)
    parser.add_argument('--valid-num-workers',     default=10, type=int)

    # Model parameters
    parser.add_argument('--model',                 default='Sequence_SkipHidden_Unet_ALL',  type=str, help='model name')    
    parser.add_argument('--loss',                  default='Sequence_SkipHidden_Unet_loss', type=str, help='loss name')    
    parser.add_argument('--method',                default='', help='multi-task weighting name')

    # Optimizer parameters
    parser.add_argument('--optimizer',             default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "AdamW"')
    
    # Learning rate and schedule and Epoch parameters
    parser.add_argument('--scheduler',             default='poly_lr', type=str, metavar='scheduler', help='scheduler (default: "poly_learning_rate"')
    parser.add_argument('--epochs',                default=1000, type=int, help='Upstream 1000 epochs, Downstream 500 epochs')  
    parser.add_argument('--warmup-epochs',         default=10, type=int, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--lr',                    default=5e-4, type=float, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--min-lr',                default=1e-5, type=float, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode',        default='DataParallel', choices=['Single', 'DataParallel'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',                default='cuda', help='device to use for training / testing')
    
    # Validation setting
    parser.add_argument('--print-freq',            default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save-checkpoint-every', default=1,  type=int, help='save the checkpoints every n epochs')  

    # Prediction and Save setting
    parser.add_argument('--checkpoint-dir',        default='', help='path where to save checkpoint or output')
    parser.add_argument('--save-dir',              default='', help='path where to prediction PNG save')

    # Continue Training
    parser.add_argument('--from-pretrained',       default='',  help='pre-trained from checkpoint')
    parser.add_argument('--resume',                default='',  help='resume from checkpoint')  # '' = None

    # Memo
    parser.add_argument('--memo',                  default='', help='memo for script')

    return parser

    
# fix random seeds for reproducibility
random_seed = 2024
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def main(args):
    start_epoch = 0
    utils.print_args(args)
    device = torch.device(args.device)
    print("cpu == ", os.cpu_count())

    # Dataloader
    data_loader_train, data_loader_valid = get_train_dataloader(name=args.dataset, args=args)   

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

    # Optimizer & LR Schedule
    if args.model == 'WGAN_VGG' or args.model == 'MAP_NN' or args.model == 'MTD_GAN' or args.model == 'Ablation_CLS' or args.model == 'Ablation_SEG' or args.model == 'Ablation_CLS_SEG' or args.model == 'Ablation_CLS_REC' or args.model == 'Ablation_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC_NDS' or args.model == 'Ablation_CLS_SEG_REC_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC_ResFFT' or args.model == 'MTD_GAN_All_One' or args.model == 'MTD_GAN_Method':
        if (args.method) and (not args.resume):
            # Weight Method such as PCGrad, CAGrad, MGDA (Ref: https://github.com/AvivNavon/nash-mtl/tree/7cc1694a276ca6f2f9426ab18b8698c786bff4f0)
            weight_methods_parameters = defaultdict(dict)
            weight_methods_parameters.update(dict(nashmtl=dict(update_weights_every=1, optim_niter=20), stl=dict(main_task=0), cagrad=dict(c=0.4), dwa=dict(temp=2.0)))
            weight_method_D = WeightMethods(method=args.method, n_tasks=3, device=device, **weight_methods_parameters[args.method])
            optimizer_D = torch.optim.AdamW([
                dict(params=model.Discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False),
                dict(params=weight_method_D.parameters(),     lr=0.025,   betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)])
            scheduler_D = get_scheduler(name=args.scheduler, optimizer=optimizer_D, args=args)
            optimizer_G = get_optimizer(name=args.optimizer, model=model.Generator, args=args)
            scheduler_G = get_scheduler(name=args.scheduler, optimizer=optimizer_G, args=args)
        else: 
            weight_method_D = None
            optimizer_D = get_optimizer(name=args.optimizer, model=model.Discriminator, args=args)
            scheduler_D = get_scheduler(name=args.scheduler, optimizer=optimizer_D, args=args)
            optimizer_G = get_optimizer(name=args.optimizer, model=model.Generator, args=args)
            scheduler_G = get_scheduler(name=args.scheduler, optimizer=optimizer_G, args=args)
    elif args.model == 'DU_GAN':
        optimizer_Img_D  = get_optimizer(name=args.optimizer,model=model.Image_Discriminator, args=args)
        scheduler_Img_D  = get_scheduler(name=args.scheduler, optimizer=optimizer_Img_D, args=args)
        optimizer_Grad_D = get_optimizer(name=args.optimizer,model=model.Grad_Discriminator, args=args)
        scheduler_Grad_D = get_scheduler(name=args.scheduler, optimizer=optimizer_Grad_D, args=args)
        optimizer_G      = get_optimizer(name=args.optimizer,model=model.Generator, args=args)
        scheduler_G      = get_scheduler(name=args.scheduler, optimizer=optimizer_G, args=args)        
    else : 
        optimizer = get_optimizer(name=args.optimizer,model=model, args=args)
        scheduler = get_scheduler(name=args.scheduler, optimizer=optimizer, args=args)

    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint['model_state_dict'] = {k.replace('.module', ''):v for k,v in checkpoint['model_state_dict'].items()} # fix loading multi-gpu 
        model.load_state_dict(checkpoint['model_state_dict'])   
        start_epoch = checkpoint['epoch'] + 1 

        if args.model == 'WGAN_VGG' or args.model == 'MAP_NN' or args.model == 'MTD_GAN' or args.model == 'Ablation_CLS' or args.model == 'Ablation_SEG' or args.model == 'Ablation_CLS_SEG' or args.model == 'Ablation_CLS_REC' or args.model == 'Ablation_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC_NDS' or args.model == 'Ablation_CLS_SEG_REC_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC_ResFFT' or args.model == 'MTD_GAN_All_One' or args.model == 'MTD_GAN_Method':
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            scheduler_D.load_state_dict(checkpoint['scheduler_D'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            scheduler_G.load_state_dict(checkpoint['scheduler_G'])
            utils.fix_optimizer(optimizer_D) # Optimizer Error fix...!
            utils.fix_optimizer(optimizer_G)
        elif args.model == 'DU_GAN':
            optimizer_Img_D.load_state_dict(checkpoint['optimizer_Img_D'])
            scheduler_Img_D.load_state_dict(checkpoint['scheduler_Img_D'])
            optimizer_Grad_D.load_state_dict(checkpoint['optimizer_Grad_D'])
            scheduler_Grad_D.load_state_dict(checkpoint['scheduler_Grad_D'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            scheduler_G.load_state_dict(checkpoint['scheduler_G'])    
            utils.fix_optimizer(optimizer_Img_D)
            utils.fix_optimizer(optimizer_Grad_D)
            utils.fix_optimizer(optimizer_G)
        else :
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            utils.fix_optimizer(optimizer)

    # Etc traing setting
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # Whole Loop Train & Valid 
    for epoch in range(start_epoch, args.epochs):

        # CNN based
            # Previous
        if args.model == 'RED_CNN' or args.model == 'ED_CNN': 
            train_stats = train_CNN_Based_Previous(model, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_CNN_Based_Previous(model, loss, data_loader_valid, device, epoch, args.save_dir, args.print_freq)
            print("Averaged valid_stats: ", valid_stats)

            # Transformer based
        elif args.model == 'Restormer' or args.model == 'CTformer': 
            train_stats = train_TR_Based_Previous(model, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_TR_Based_Previous(model, loss, data_loader_valid, device, epoch, args.save_dir, args.print_freq)
            print("Averaged valid_stats: ", valid_stats)

        # GAN based
            # Previous        
        elif args.model == 'WGAN_VGG': 
            train_stats = train_WGAN_VGG_Previous(model, data_loader_train, optimizer_G, optimizer_D, device, epoch, args.print_freq, args.batch_size)            
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_WGAN_VGG_Previous(model, loss, data_loader_valid, device, epoch, args.save_dir, args.print_freq)
            print("Averaged valid_stats: ", valid_stats)


        elif args.model == 'MAP_NN': 
            train_stats = train_MAP_NN_Previous(model, data_loader_train, optimizer_G, optimizer_D, device, epoch, args.print_freq, args.batch_size)            
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_MAP_NN_Previous(model, loss, data_loader_valid, device, epoch, args.save_dir, args.print_freq)
            print("Averaged valid_stats: ", valid_stats)


        elif args.model == 'DU_GAN': 
            train_stats = train_DUGAN_Previous(model, data_loader_train, optimizer_G, optimizer_Img_D, optimizer_Grad_D, device, epoch, args.print_freq, args.batch_size)            
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_DUGAN_Previous(model, loss, data_loader_valid, device, epoch, args.save_dir, args.print_freq)
            print("Averaged valid_stats: ", valid_stats)


        # DN based
            # Previous        
        elif args.model == 'DDPM' or args.model == 'DDIM' or args.model == 'PNDM' or args.model == 'DPM':
            train_stats = train_DN_Previous(model, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_DN_Previous(model, loss, data_loader_valid, device, epoch, args.save_dir, args.print_freq)
            print("Averaged valid_stats: ", valid_stats)


            # Ours
        elif args.model == 'MTD_GAN' or args.model == 'Ablation_CLS' or args.model == 'Ablation_SEG' or args.model == 'Ablation_CLS_SEG' or args.model == 'Ablation_CLS_REC' or args.model == 'Ablation_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC_NDS' or args.model == 'Ablation_CLS_SEG_REC_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC_ResFFT' or args.model == 'MTD_GAN_All_One' or args.model == 'MTD_GAN_Method':
            train_stats = train_MTD_GAN_Ours(model, data_loader_train, optimizer_G, optimizer_D, device, epoch, args.print_freq, args.batch_size, weight_method_D)
            print("Averaged train_stats: ", train_stats)
            valid_stats = valid_MTD_GAN_Ours(model, loss, data_loader_valid, device, epoch, args.save_dir, args.print_freq)
            print("Averaged valid_stats: ", valid_stats)
        
        # LR scheduler update
        if args.model == 'WGAN_VGG' or args.model == 'MAP_NN' or args.model == 'MTD_GAN' or args.model == 'Ablation_CLS' or args.model == 'Ablation_SEG' or args.model == 'Ablation_CLS_SEG' or args.model == 'Ablation_CLS_REC' or args.model == 'Ablation_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC_NDS' or args.model == 'Ablation_CLS_SEG_REC_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC_ResFFT' or args.model == 'MTD_GAN_All_One' or args.model == 'MTD_GAN_Method':
            scheduler_G.step(epoch)
            scheduler_D.step(epoch)
        elif args.model == 'DU_GAN':            
            scheduler_G.step(epoch)
            scheduler_Img_D.step(epoch)
            scheduler_Grad_D.step(epoch)
        else:    
            scheduler.step(epoch)            

        # Save checkpoint & Prediction png
        if epoch % args.save_checkpoint_every == 0:
            checkpoint_path = args.checkpoint_dir + '/epoch_' + str(epoch) + '_checkpoint.pth'
            
            if args.model == 'WGAN_VGG' or args.model == 'MAP_NN' or args.model == 'MTD_GAN' or args.model == 'Ablation_CLS' or args.model == 'Ablation_SEG' or args.model == 'Ablation_CLS_SEG' or args.model == 'Ablation_CLS_REC' or args.model == 'Ablation_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC' or args.model == 'Ablation_CLS_SEG_REC_NDS' or args.model == 'Ablation_CLS_SEG_REC_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC' or args.model == 'Ablation_CLS_SEG_REC_NDS_RC_ResFFT' or args.model == 'MTD_GAN_All_One' or args.model == 'MTD_GAN_Method':
                torch.save({
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                    'optimizer_D': optimizer_D.state_dict(),
                    'scheduler_D': scheduler_D.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'scheduler_G': scheduler_G.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
           
            elif args.model == 'DU_GAN':
                torch.save({
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                    'optimizer_Img_D': optimizer_Img_D.state_dict(),
                    'scheduler_Img_D': scheduler_Img_D.state_dict(),
                    'optimizer_Grad_D': optimizer_Grad_D.state_dict(),
                    'scheduler_Grad_D': scheduler_Grad_D.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'scheduler_G': scheduler_G.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)   

            else :
                torch.save({
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),  # Save only Single Gpu mode
                    'optimizer': optimizer.state_dict(), 
                    'scheduler': scheduler.state_dict(), 
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)                                        

        # Log & Save
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'valid_{k}': v for k, v in valid_stats.items()},
                     'epoch': epoch}

        with open(args.checkpoint_dir + "/log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # Finish
    total_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MTD-GAN training script', parents=[get_args_parser()])
    args = parser.parse_args()

    # Make folder if not exist
    os.makedirs(args.checkpoint_dir + "/args", mode=0o777, exist_ok=True)
    os.makedirs(args.save_dir, mode=0o777, exist_ok=True)
    
    # Save args to json
    if not os.path.isfile(args.checkpoint_dir + "/args/args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json"):
        with open(args.checkpoint_dir + "/args/args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    main(args)



