import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import argparse
from loader_2d import get_loader
from prep import printProgressBar
from networks import UNet
from measure import compute_measure
import numpy as np
import torchvision.utils as utils

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--load_mode', type=int, default=1)
parser.add_argument('--data_path', type=str, default='/home/sutanu/CT_d/CT')
parser.add_argument('--saved_path', type=str, default='/home/sutanu/CT_d/data/npy_img/')
parser.add_argument('--save_path', type=str, default='/home/sutanu/CT_d/Checkpoint/save/')
parser.add_argument('--test_patient', type=str, default='L506')
parser.add_argument('--result_fig', type=bool, default=True)

parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=3072.0)
parser.add_argument('--trunc_min', type=float, default=-160.0)
parser.add_argument('--trunc_max', type=float, default=240.0)

parser.add_argument('--transform', type=bool, default=False)
# if patch training, batch size is (--patch_n * --batch_size)
parser.add_argument('--patch_n', type=int, default=8)
parser.add_argument('--patch_size', type=int, default=120)
parser.add_argument('--batch_size', type=int, default=4)





parser.add_argument('--num_workers', type=int, default=7)









args = parser.parse_args()

data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

def denormalize_(image):
    image = image * (args.norm_range_max - args.norm_range_min) + args.norm_range_min
    return image

def normalize_(image, MIN_B=-160.0, MAX_B=240.0):
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return image

def trunc(mat):
    mat[mat <= args.trunc_min] = args.trunc_min
    mat[mat >= args.trunc_max] = args.trunc_max
    return mat


def save_fig(x, y, pred, fig_name, original_result, pred_result):
    x, y, pred = np.squeeze(x), np.squeeze(y), np.squeeze(pred)
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(x, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                       original_result[1],
                                                                       original_result[2]), fontsize=20)
    ax[1].imshow(pred, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                       pred_result[1],
                                                                       pred_result[2]), fontsize=20)
    ax[2].imshow(y, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
    ax[2].set_title('Full-dose', fontsize=30)

    f.savefig(os.path.join(args.save_path, 'fig', 'result_{}.png'.format(fig_name)))
    plt.close()
# load
    
    
whole_model = torch.load(args.save_path + 'latest_ckpt.pth.tar')
netG_state_dict= whole_model['netG_state_dict']
netG = UNet()
netG = netG.cuda()
netG.load_state_dict(netG_state_dict)


# compute PSNR, SSIM, RMSE
ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

k=0
with torch.no_grad():
    for i, (x, y) in enumerate(data_loader):

        shape_ = x.shape[-1]
        x = x.unsqueeze(0).float().cuda()
        y = y.unsqueeze(0).float().cuda()

        pred = netG(x)

        # denormalize, truncate
        x = trunc(denormalize_(x.view(shape_, shape_).cpu().detach()))
        y = trunc(denormalize_(y.view(shape_, shape_).cpu().detach()))
        pred = trunc(denormalize_(pred.view(shape_, shape_).cpu().detach()))

        data_range = args.trunc_max - args.trunc_min

        original_result, pred_result = compute_measure(x, y, pred, data_range)

        ori_psnr_avg += original_result[0]
        ori_ssim_avg += original_result[1]
        ori_rmse_avg += original_result[2]
        pred_psnr_avg += pred_result[0]
        pred_ssim_avg += pred_result[1]
        pred_rmse_avg += pred_result[2]

        # save result figure
        if args.result_fig:
            save_fig(x, y, pred, i, original_result, pred_result)
            pred=normalize_(pred.numpy())
            pred=torch.Tensor(pred)
            utils.save_image(pred, os.path.join(args.save_path, 'fig', 'Pred_{}.png'.format(i)))

        printProgressBar(i, len(data_loader),
                         prefix="Compute measurements ..",
                         suffix='Complete', length=25)
    print('\n')
    print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(data_loader), 
                                                                                    ori_ssim_avg/len(data_loader), 
                                                                                    ori_rmse_avg/len(data_loader)))
    print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(data_loader), 
                                                                                          pred_ssim_avg/len(data_loader), 
                                                                                          pred_rmse_avg/len(data_loader)))