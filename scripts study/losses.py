import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from torch.nn.modules.loss import _Loss
from module.loss import MS_SSIM_L1_LOSS

# Vgg multiple loss Ref : https://github.com/NVIDIA/pix2pixHD/blob/5a2c87201c5957e2bf51d79b8acddb9cc1920b26/models/networks.py#L112
# Resnet loss ref : https://github.com/workingcoder/EDCNN

class HF_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        super(HF_Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.kernel_weight = nn.Parameter(torch.zeros(size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        for idx in range(out_channels):
            
            # Sobel Filter
            if idx == 0:
                self.kernel_weight[idx, :, 0, :] = -1          
                self.kernel_weight[idx, :, 0, 1] = -2
                self.kernel_weight[idx, :, -1, :] = 1
                self.kernel_weight[idx, :, -1, 1] = 2
            elif idx == 1:
                self.kernel_weight[idx, :, :, 0] = -1
                self.kernel_weight[idx, :, 1, 0] = -2
                self.kernel_weight[idx, :, :, -1] = 1
                self.kernel_weight[idx, :, 1, -1] = 2
            elif idx == 2:
                self.kernel_weight[idx, :, 0, 0] = -2
                self.kernel_weight[idx, :, 0, 1] = -1
                self.kernel_weight[idx, :, 1, 0] = -1
                self.kernel_weight[idx, :, 1, -1] = 1
                self.kernel_weight[idx, :, -1, 1] = 1
                self.kernel_weight[idx, :, -1, -1] = 2
            elif idx == 3:
                self.kernel_weight[idx, :, 0, 1] = 1
                self.kernel_weight[idx, :, 0, -1] = 2
                self.kernel_weight[idx, :, 1, 0] = -1
                self.kernel_weight[idx, :, 1, -1] = 1
                self.kernel_weight[idx, :, -1, 0] = -2
                self.kernel_weight[idx, :, -1, 1] = -1

            # High Frequency (Image - Blur)
            elif idx == 4:
                self.kernel_weight[idx, :, :, :] = -1/16
                self.kernel_weight[idx, :, 1, :] = -2/16
                self.kernel_weight[idx, :, :, 1] = -2/16
                self.kernel_weight[idx, :, 1, 1] = 12/16      

            # Laplacian or Unsharped mask filter or point edge filter
            elif idx == 5:
                self.kernel_weight[idx, :, 1, :] = -1           
                self.kernel_weight[idx, :, :, 1] = -1
                self.kernel_weight[idx, :, 1, 1] = 4
            elif idx == 6:
                self.kernel_weight[idx, :, :, :] = -1           
                self.kernel_weight[idx, :, 1, 1] += 9   
            elif idx == 7:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 1, :] = -2
                self.kernel_weight[idx, :, :, 1] = -2
                self.kernel_weight[idx, :, 1, 1] = 4

            # Compass Prewitt
            elif idx == 8:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 0, :] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 9:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 0:2, 1:3] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 10:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, :, 2] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 11:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 1:3, 1:3] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 12:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 1, 1] = -2
                self.kernel_weight[idx, :, 2, :] = -1    
            elif idx == 13:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, 1:3, 0:2] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 14:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, :, 0] = -1
                self.kernel_weight[idx, :, 1, 1] = -2
            elif idx == 15:
                self.kernel_weight[idx, :, :, :] = 1           
                self.kernel_weight[idx, :, :2, :2] = -1
                self.kernel_weight[idx, :, 1, 1] = -2

            # Line filter
            elif idx == 16:
                self.kernel_weight[idx, :, :, :] = -1           
                self.kernel_weight[idx, :, 1, :] = 2
            elif idx == 17:
                self.kernel_weight[idx, :, :, :] = -1           
                self.kernel_weight[idx, :, :, 1] = 2
            elif idx == 18:
                self.kernel_weight[idx, :, :, :] = -1           
                self.kernel_weight[idx, :, 0, 2] = 2
                self.kernel_weight[idx, :, 1, 1] = 2
                self.kernel_weight[idx, :, 2, 0] = 2
            elif idx == 19:
                self.kernel_weight[idx, :, :, :] = -1           
                self.kernel_weight[idx, :, 0, 0] = 2
                self.kernel_weight[idx, :, 1, 1] = 2
                self.kernel_weight[idx, :, 2, 2] = 2
                                             

        # Define the trainable sobel factor
        if requires_grad:
            self.kernel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32), requires_grad=True)
        else:
            self.kernel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        # if torch.cuda.is_available():
        #     self.kernel_factor = self.kernel_factor.cuda()
        #     if isinstance(self.bias, nn.Parameter):
        #         self.bias = self.bias.cuda()

        kernel_weight = self.kernel_weight * self.kernel_factor

        # if torch.cuda.is_available():
        #     kernel_weight = kernel_weight.cuda()

        out = F.conv2d(x, kernel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # return torch.cat([out, x], dim=1)
        return out

class LF_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        super(LF_Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.kernel_weight = nn.Parameter(torch.zeros(size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        for idx in range(out_channels):
            
            # Box Blur
            if idx == 0:
                self.kernel_weight[idx, :, :, :] = 1/9

            # Gaussian Blur
            elif idx == 1:
                self.kernel_weight[idx, :, :, :] = 1/16
                self.kernel_weight[idx, :, 1, :] = 2/16
                self.kernel_weight[idx, :, :, 1] = 2/16
                self.kernel_weight[idx, :, 1, 1] = 4/16      

        # Define the trainable sobel factor
        if requires_grad:
            self.kernel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32), requires_grad=True)
        else:
            self.kernel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        kernel_weight = self.kernel_weight * self.kernel_factor
        out = F.conv2d(x, kernel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return torch.cat([out, x], dim=1)

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
class VGGLoss(torch.nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().to(device)
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):   
        self.vgg.eval()
        with torch.no_grad():                   
            x_vgg, y_vgg = self.vgg(x.repeat(1,3,1,1)), self.vgg(y.repeat(1,3,1,1))
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i]*self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class VGG_Triple_Loss(torch.nn.Module):
    def __init__(self, device):
        super(VGG_Triple_Loss, self).__init__()        
        self.vgg = Vgg19().to(device)
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, anchor, positive, negative):              
        self.vgg.eval()
        with torch.no_grad():
            a_vgg, p_vgg, n_vgg = self.vgg(anchor.repeat(1,3,1,1)), self.vgg(positive.repeat(1,3,1,1)), self.vgg(negative.repeat(1,3,1,1))
            
        loss = 0
        margin = nn.functional.l1_loss(input=negative, target=positive)

        assert len(a_vgg) == len(p_vgg) == len(n_vgg)
        for i in range(len(a_vgg)):
            loss += self.weights[i]*nn.functional.triplet_margin_loss(anchor=a_vgg[i].flatten(1), positive=p_vgg[i].detach().flatten(1), negative=n_vgg[i].flatten(1), margin=margin, p=1)        
            
        return loss

class ResNet50FeatureExtractor(nn.Module):

    def __init__(self, blocks=[1, 2, 3, 4], pretrained=False, progress=True, **kwargs):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained, progress, **kwargs)
        del self.model.avgpool
        del self.model.fc
        self.blocks = blocks

    def forward(self, x):
        feats = list()

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if 1 in self.blocks:
            feats.append(x)

        x = self.model.layer2(x)
        if 2 in self.blocks:
            feats.append(x)

        x = self.model.layer3(x)
        if 3 in self.blocks:
            feats.append(x)

        x = self.model.layer4(x)
        if 4 in self.blocks:
            feats.append(x)

        return feats

class Window_Conv2D(nn.Module):
    '''
    HU summary  
          [HU threshold]                                 [0 ~ 1 Range]                 [weight / bias]
    brain          = W:80 L:40                          W:0.250 L:0.270               W:50.000 B:-12.500
    subdural       = W:130-300 L:50-100                 W:0.246 L:0.278               W:31.250 B:-7.687
    stroke         = W:8 L:32 or W:40 L:40              W:0.257 L:0.259               W:45.455 B:-11.682
    temporal bones = W:2800 L:600 or W:4000 L:700       W:0.055 L:0.738               W:1.464  B:-0.081
    soft tisuues   = W:350-400 L:20-60                  W:0.212 L:0.298               W:11.628 B:-2.465
    '''        
    def __init__(self, mode, in_channels=1, out_channels=5):
        super(Window_Conv2D, self).__init__()
        self.out_channels = out_channels
        self.conv_layer   = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        
        if mode == "relu":
            self.act_layer = self.upbound_relu
        elif mode == "sigmoid":
            self.act_layer = self.upbound_sigmoid
        else:
            raise Exception()
        
        # Initialize by xavier_uniform_
        self.init_weight()
        
    def upbound_relu(self, x):
        return torch.minimum(torch.maximum(x, torch.tensor(0)), torch.tensor(1.0))

    def upbound_sigmoid(self, x):
        return 1.0 * torch.sigmoid(x)
                    
    def init_weight(self):
        print("inintializing...!")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):        
                for idx in range(self.out_channels):
                    if idx % 5 == 0:                       
                        nn.init.constant_(m.weight[0, :, :, :], 50.0)    # torch.Size([5, 1, 1, 1])
                        nn.init.constant_(m.bias[0], -12.5)              # torch.Size([5])                  
                    elif idx % 5 == 1:                       
                        nn.init.constant_(m.weight[1, :, :, :], 31.250)  # torch.Size([5, 1, 1, 1])
                        nn.init.constant_(m.bias[1], -7.687)             # torch.Size([5])                  
                    elif idx % 5 == 2:                       
                        nn.init.constant_(m.weight[2, :, :, :], 45.455)  # torch.Size([5, 1, 1, 1])
                        nn.init.constant_(m.bias[2], -11.682)            # torch.Size([5])                  
                    elif idx % 5 == 3:                       
                        nn.init.constant_(m.weight[3, :, :, :], 1.464)   # torch.Size([5, 1, 1, 1])
                        nn.init.constant_(m.bias[3], -0.081)             # torch.Size([5])                  
                    elif idx % 5 == 4:                       
                        nn.init.constant_(m.weight[4, :, :, :], 11.628)  # torch.Size([5, 1, 1, 1])
                        nn.init.constant_(m.bias[4], -2.465)             # torch.Size([5])                  
                    else :                       
                        raise Exception()
                                     
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.act_layer(x)
        return x
    
    def inference(self, x):
        self.eval()
        with torch.no_grad():
            x = self.conv_layer(x)
            x = self.act_layer(x)
        return x    

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

######################################################################################################################################################################
######################################################             LOSS           Class                      ########################################################
######################################################################################################################################################################

class Perceptual_L1_Loss(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.loss_L1      = torch.nn.L1Loss()
        self.loss_VGG     = VGGLoss(device='cuda')
        self.mode = mode

    def forward(self, pred_n_40=None, pred_n_60=None, pred_n_80=None, pred_n_100=None, gt_20=None, gt_40=None, gt_60=None, gt_80=None, gt_100=None):
        if self.mode == "multi_label" :

            loss_n_40  = self.loss_VGG(pred_n_40,  gt_40)  + self.loss_L1(pred_n_40,  gt_40)
            loss_n_60  = self.loss_VGG(pred_n_60,  gt_60)  + self.loss_L1(pred_n_60,  gt_60)
            loss_n_80  = self.loss_VGG(pred_n_80,  gt_80)  + self.loss_L1(pred_n_80,  gt_80)
            loss_n_100 = self.loss_VGG(pred_n_100, gt_100) + self.loss_L1(pred_n_100, gt_100)

            total_loss = loss_n_40 + loss_n_60 + loss_n_80 + loss_n_100

            return total_loss
        
        else :

            loss_n_100 = self.loss_VGG(pred_n_100, gt_100) + self.loss_L1(pred_n_100, gt_100)

            return loss_n_100

class Perceptual_L1_MS_SSIM_Loss(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.loss_MS_SSIM_L1  = MS_SSIM_L1_LOSS(gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0], data_range=1.0, K=(0.01, 0.03), alpha=0.84, compensation=200.0)
        self.loss_VGG         = VGGLoss(device='cuda')
        self.mode = mode

    def forward(self, pred_n_40=None, pred_n_60=None, pred_n_80=None, pred_n_100=None, gt_20=None, gt_40=None, gt_60=None, gt_80=None, gt_100=None):

        if self.mode == "multi_label" :
            
            loss_n_40  = self.loss_VGG(pred_n_40,  gt_40)  + self.loss_MS_SSIM_L1(pred_n_40,  gt_40)
            loss_n_60  = self.loss_VGG(pred_n_60,  gt_60)  + self.loss_MS_SSIM_L1(pred_n_60,  gt_60)
            loss_n_80  = self.loss_VGG(pred_n_80,  gt_80)  + self.loss_MS_SSIM_L1(pred_n_80,  gt_80)
            loss_n_100 = self.loss_VGG(pred_n_100, gt_100) + self.loss_MS_SSIM_L1(pred_n_100, gt_100)

            total_loss = loss_n_40 + loss_n_60 + loss_n_80 + loss_n_100

            return total_loss, {'loss_n_40':loss_n_40.item(), 'loss_n_60':loss_n_60.item(), 'loss_n_80':loss_n_80.item(), 'loss_n_100':loss_n_100.item()}
        
        else:
            loss_n_100 = self.loss_VGG(pred_n_100, gt_100) + self.loss_MS_SSIM_L1(pred_n_100, gt_100)

            return loss_n_100, {'loss_n_100':loss_n_100.item()}

class Perceptual_Triple_L1_Loss(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.loss_VGG_Triple  = VGG_Triple_Loss(device='cuda') # shape must be (B, C)
        self.loss_L1          = torch.nn.L1Loss()
        
        self.mode = mode

    def forward(self, pred_n_40=None, pred_n_60=None, pred_n_80=None, pred_n_100=None, gt_20=None, gt_40=None, gt_60=None, gt_80=None, gt_100=None):
        
        if self.mode == "multi_label" :
            
            loss_n_40  = self.loss_VGG_Triple(anchor=pred_n_100, positive=gt_100, negative=gt_20) + self.loss_L1(pred_n_100,  gt_100)
            loss_n_60  = self.loss_VGG_Triple(anchor=pred_n_100, positive=gt_100, negative=gt_20) + self.loss_L1(pred_n_100,  gt_100)
            loss_n_80  = self.loss_VGG_Triple(anchor=pred_n_100, positive=gt_100, negative=gt_20) + self.loss_L1(pred_n_100,  gt_100)
            loss_n_100 = self.loss_VGG_Triple(anchor=pred_n_100, positive=gt_100, negative=gt_20) + self.loss_L1(pred_n_100,  gt_100)

            total_loss = loss_n_40 + loss_n_60 + loss_n_80 + loss_n_100

            return total_loss, {'loss_n_40':loss_n_40.item(), 'loss_n_60':loss_n_60.item(), 'loss_n_80':loss_n_80.item(), 'loss_n_100':loss_n_100.item()}
        
        else:
            loss_n_100 = self.loss_VGG_Triple(anchor=pred_n_100, positive=gt_100, negative=gt_20) + self.loss_L1(pred_n_100,  gt_100)

            return loss_n_100, {'loss_n_100':loss_n_100.item()}

class CompoundLoss(_Loss):
    def __init__(self, blocks=[1, 2, 3, 4], mse_weight=1.0, resnet_weight=0.01):
        super(CompoundLoss, self).__init__()

        self.mse_weight    = mse_weight
        self.resnet_weight = resnet_weight

        self.blocks = blocks
        self.resnet = ResNet50FeatureExtractor(pretrained=True)

        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()
        self.resnet.eval()

        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        loss_value = 0

        input_feats  = self.resnet(torch.cat([input, input, input], dim=1))
        target_feats = self.resnet(torch.cat([target, target, target], dim=1))

        feats_num = len(self.blocks)
        for idx in range(feats_num):
            loss_value += self.criterion(input_feats[idx], target_feats[idx])
        loss_value /= feats_num

        loss = self.mse_weight*self.criterion(input, target) + self.resnet_weight*loss_value

        return loss

class Window_CompoundLoss(_Loss):
    def __init__(self, blocks=[1, 2, 3, 4], mse_weight=0.5, resnet_weight=0.01, window_weight=0.5):
        super(Window_CompoundLoss, self).__init__()
        self.window_conv   = Window_Conv2D(mode='relu', in_channels=1, out_channels=5)
        self.mse_weight    = mse_weight
        self.resnet_weight = resnet_weight
        self.window_weight = window_weight

        self.blocks = blocks
        self.resnet = ResNet50FeatureExtractor(pretrained=True)

        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()
        self.resnet.eval()

        self.criterion = nn.MSELoss()
        self.window_conv.cuda()

    def forward(self, input, target):
        loss_value = 0

        input_feats  = self.resnet(torch.cat([input, input, input], dim=1))
        target_feats = self.resnet(torch.cat([target, target, target], dim=1))

        feats_num = len(self.blocks)
        for idx in range(feats_num):
            loss_value += self.criterion(input_feats[idx], target_feats[idx])
        loss_value /= feats_num
        
        loss = self.mse_weight*self.criterion(input, target) + self.resnet_weight*loss_value + self.window_weight*self.criterion(self.window_conv(input), self.window_conv(target))

        return loss

class Change_L2_L1_Loss(_Loss):
    def __init__(self, change_epoch=10):
        super(Change_L2_L1_Loss, self).__init__()
        self.change_epoch  = change_epoch

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
    def forward(self, input, target, epoch):

        if epoch > self.change_epoch:
            loss = self.l1_loss(target, input) * epoch

        else: 
            print("Still L2 Loss...!")
            loss = self.l2_loss(target, input)

        return loss

class Window_L1_Loss(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.window_conv  = Window_Conv2D(mode='relu', in_channels=1, out_channels=5)
        self.loss_L1      = torch.nn.L1Loss()
        
        self.mode = mode
        self.window_conv.cuda()

    def forward(self, gt_high, target):
        # print(self.window_conv.inference(gt_high).shape)
        # print(self.window_conv.inference(gt_high))
        # Window
        window_loss  = self.loss_L1(self.window_conv.inference(gt_high), self.window_conv.inference(target))

        # L1 
        l1_loss      = self.loss_L1(target,  gt_high)

        return window_loss + l1_loss

class Charbonnier_HighFreq_Loss(nn.Module):
    def __init__(self, eps=1e-3):
        super(Charbonnier_HighFreq_Loss, self).__init__()
        self.eps     = eps
        self.HF_conv = HF_Conv(in_channels=1, out_channels=20, requires_grad=False).to('cuda')
        self.L1      = torch.nn.L1Loss()

    def forward(self, gt_100, pred_n_100):
        # Charbonnier Loss
        diff  = gt_100 - pred_n_100
        loss1 = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))

        # High Freq Loss
        loss2 = self.L1(self.HF_conv(gt_100), self.HF_conv(pred_n_100))

        return loss1 + loss2*100.0, {'Charbonnier_Loss': loss1, 'HF_Loss': loss2}




def create_criterion(name, mode):
    #### Select Loss
    if name == 'Perceptual+L1 Loss':
        criterion = Perceptual_L1_Loss(mode=mode)

    elif name == 'Perceptual+L1+MS_SSIM Loss':
        criterion = Perceptual_L1_MS_SSIM_Loss(mode=mode)        

    elif name == 'L2 Loss':
        criterion = nn.MSELoss()

    elif name == 'L1 Loss':
        criterion = nn.L1Loss()        
        
    elif name == 'Perceptual_Triple+L1_Loss':
        criterion = Perceptual_Triple_L1_Loss(mode=mode)        

    elif name == 'Compound Loss':  # L2 + ResNet50 loss
        criterion = CompoundLoss(blocks=[1, 2, 3, 4], mse_weight=1, resnet_weight=0.01)

    elif name == 'Window Compound Loss':  # L2 + ResNet50 loss + Window L2
        criterion = Window_CompoundLoss(blocks=[1, 2, 3, 4], mse_weight=0.1, resnet_weight=0.01, window_weight=0.9)

    elif name == 'Window L1 Loss':  # L2 + ResNet50 loss + Window L2
        criterion = Window_L1_Loss(mode=mode)        

    elif name == 'Change L2 L1 Loss':  # L2 + ResNet50 loss + Window L2
        criterion = Change_L2_L1_Loss(change_epoch=10)

    elif name == 'Charbonnier_HighFreq_Loss':  # L2 + ResNet50 loss + Window L2
        criterion = Charbonnier_HighFreq_Loss()

    else: 
        raise Exception('Error...! name')

    return criterion



