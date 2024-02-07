import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torchvision import models

# Vgg multiple loss Ref : https://github.com/NVIDIA/pix2pixHD/blob/5a2c87201c5957e2bf51d79b8acddb9cc1920b26/models/networks.py#L112
# Resnet loss       Ref : https://github.com/workingcoder/EDCNN


def ls_gan(inputs, targets):
    return torch.mean((inputs - targets) ** 2)

def NDS_Loss(inputs, targets, diffs):
    # Non-difference suppression loss from LSGAN
    return torch.mean( torch.abs(diffs).bool() * (inputs - targets)**2 )

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

class ResNet50FeatureExtractor(torch.nn.Module):
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

class CharbonnierLoss(torch.nn.Module):
    """
    Charbonnier Loss (L1)
    ref: https://github.com/swz30/MPRNet/blob/51b58bb2ec803162e9053c1269b170009ee6f693/Deblurring/losses.py
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,1,1)  # 1 -> gray channel
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class MSFRLoss(torch.nn.Module):
    def __init__(self):
        super(MSFRLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):

        x_fft = torch.fft.rfftn(x)
        y_fft = torch.fft.rfftn(y)
      
        loss = self.l1_loss(x_fft, y_fft)
        
        return loss

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


def get_loss(name):

    if name == 'L2 Loss':
        criterion = torch.nn.MSELoss()

    elif name == 'L1 Loss':
        criterion = torch.nn.L1Loss()        

    else: 
        raise Exception('Error...! name')

    return criterion



