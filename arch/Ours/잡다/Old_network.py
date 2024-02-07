import numpy as np
import skimage
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_model
from torch.autograd import Variable


from .cmt_module import *
from .CMT import *
from .RRDBNet_arch import RRDB_Net
from .gpen_model import *
from .DiffAugment_pytorch import DiffAugment
from .Unet_Factory import Revised_UNet, DownsampleBlock, UpsampleBlock
from .DUGAN_wrapper import *
from .edcnn_model import *



############################################################################################################
# CNN - Based
############################################################################################################
# 1. CMT Unet
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu  = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))    
        return x

class OutputProj(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)

    def forward(self, x):
        x = self.conv1(x)
        return x

class CMT_Unet(nn.Module):
    def __init__(self):
        super(CMT_Unet, self).__init__()

        # CMT Encoder
        # self.encoder         = CMT_S(img_size = 64, num_class = 1000)
        self.encoder         = CMT_S(img_size = 512, num_class = 1000)

        # CMT Decoder
        self.upsample1       = UpsampleBlock(in_channels=512, up_scale=2) 
        self.decoder_block1  = ConvBlock(in_ch=512, out_ch=256)
        
        self.upsample2       = UpsampleBlock(in_channels=256, up_scale=2)         
        self.decoder_block2  = ConvBlock(in_ch=256, out_ch=128)
        
        self.upsample3       = UpsampleBlock(in_channels=128, up_scale=2) 
        self.decoder_block3  = ConvBlock(in_ch=128, out_ch=64)
        
        self.upsample4       = UpsampleBlock(in_channels=64, up_scale=2) 
        self.decoder_block4  = ConvBlock(in_ch=64, out_ch=32)
        
        self.upsample5       = UpsampleBlock(in_channels=32, up_scale=2) 
        self.decoder_block5  = ConvBlock(in_ch=32, out_ch=16)

        self.head            = OutputProj(in_channels=16, out_channels=1)
        self.relu            = nn.ReLU()

        # Windowing Downsampling Conv

        # Initialize by xavier_uniform_
        self.init_weight()
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # x shape is (B, C, H, W)
        skip4, skip3, skip2, skip1, center = self.encoder(x)
        # print(features[0].shape) torch.Size([64, 32, 32, 32])
        # print(features[1].shape) torch.Size([64, 64, 16, 16])
        # print(features[2].shape) torch.Size([64, 128, 8, 8])
        # print(features[3].shape) torch.Size([64, 256, 4, 4])
        # print(features[4].shape) torch.Size([64, 512, 2, 2])
        
        # print("4 = ", skip4.shape)    [16, 32, 256, 256]
        # print("3 = ", skip3.shape)    [16, 64, 128, 128]
        # print("2 = ", skip2.shape)    [16, 128, 64, 64]
        # print("1 = ", skip1.shape)  [16, 256, 32, 32]
        # print("0 = ", center.shape) [16, 512, 16, 16]
        out   = self.upsample1(center)
        # print("2 = ", out.shape) [16, 512, 32, 32]
        out   = self.decoder_block1(out)
        out   += skip1
        
        out   = self.upsample2(self.relu(out))
        out   = self.decoder_block2(out)
        out   += skip2

        out   = self.upsample3(self.relu(out))
        out   = self.decoder_block3(out)
        out   += skip3
        
        out   = self.upsample4(self.relu(out))
        out   = self.decoder_block4(out)
        out   += skip4
        
        out   = self.upsample5(self.relu(out))
        out   = self.decoder_block5(out)
        out   = self.head(out)

        out   += x
    
        return self.relu(out)

# 2. WDCNN
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
        out = self.conv_layer(x)
        out = self.act_layer(out)
        # out = torch.cat([x, out], dim=1)
        return out
    
    def inference(self, x):
        self.eval()
        with torch.no_grad():
            x = self.conv_layer(x)
            x = self.act_layer(x)
        return x    

class WDCNN(nn.Module):

    def __init__(self, in_ch=1, out_ch=32, window_ch=30):
        super(WDCNN, self).__init__()

        self.conv_window = Window_Conv2D(mode='relu', in_channels=in_ch, out_channels=window_ch)

        self.conv_p1 = nn.Conv2d(in_ch + window_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p2 = nn.Conv2d(in_ch + window_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p3 = nn.Conv2d(in_ch + window_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p4 = nn.Conv2d(in_ch + window_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p5 = nn.Conv2d(in_ch + window_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f5 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p6 = nn.Conv2d(in_ch + window_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f6 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p7 = nn.Conv2d(in_ch + window_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f7 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p8 = nn.Conv2d(in_ch + window_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f8 = nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out_0 = self.conv_window(x)
        out_0 = torch.cat((x, out_0), dim=-3)

        out_1 = self.relu(self.conv_p1(out_0))
        out_1 = self.relu(self.conv_f1(out_1))
        out_1 = torch.cat((out_0, out_1), dim=-3)

        out_2 = self.relu(self.conv_p2(out_1))
        out_2 = self.relu(self.conv_f2(out_2))
        out_2 = torch.cat((out_0, out_2), dim=-3)

        out_3 = self.relu(self.conv_p3(out_2))
        out_3 = self.relu(self.conv_f3(out_3))
        out_3 = torch.cat((out_0, out_3), dim=-3)

        out_4 = self.relu(self.conv_p4(out_3))
        out_4 = self.relu(self.conv_f4(out_4))
        out_4 = torch.cat((out_0, out_4), dim=-3)

        out_5 = self.relu(self.conv_p5(out_4))
        out_5 = self.relu(self.conv_f5(out_5))
        out_5 = torch.cat((out_0, out_5), dim=-3)

        out_6 = self.relu(self.conv_p6(out_5))
        out_6 = self.relu(self.conv_f6(out_6))
        out_6 = torch.cat((out_0, out_6), dim=-3)

        out_7 = self.relu(self.conv_p7(out_6))
        out_7 = self.relu(self.conv_f7(out_7))
        out_7 = torch.cat((out_0, out_7), dim=-3)

        out_8 = self.relu(self.conv_p8(out_7))
        out_8 = self.conv_f8(out_8)

        out = self.relu(x + out_8)

        return out




############################################################################################################
# GAN - Based
############################################################################################################

# 1. Main MAP_ECMT GAN Model
class Light_ECMT_2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=32, sobel_ch=32):
        super(Light_ECMT_2D, self).__init__()

        self.conv_sobel  = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_layer1 = nn.Conv2d(in_ch + sobel_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.cmt_layer1  = CMTBlock(img_size=64, stride=2, d_k=out_ch, d_v=out_ch, num_heads=4, R=4, in_channels=out_ch)

        self.conv_layer2 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.cmt_layer2  = CMTBlock(img_size=64, stride=2, d_k=out_ch, d_v=out_ch, num_heads=4, R=4, in_channels=out_ch)

        self.conv_layer3 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.cmt_layer3  = CMTBlock(img_size=64, stride=2, d_k=out_ch, d_v=out_ch, num_heads=4, R=4, in_channels=out_ch)

        self.head = nn.Conv2d(in_ch + sobel_ch + out_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.relu = nn.LeakyReLU(inplace=False)

        # Initialize by xavier_uniform_
        self.init_weight()
        
    def init_weight(self):
        print("inintializing...!")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # x shape is (B, C, H, W)
        # Encoder
        out_0 = self.conv_sobel(x)
        out_0 = torch.cat((x, out_0), dim=-3)

        out_1 = self.relu(self.conv_layer1(out_0))
        out_1 = self.relu(self.cmt_layer1(out_1))
        out_1 = torch.cat((out_0, out_1), dim=-3)

        out_2 = self.relu(self.conv_layer2(out_1))
        out_2 = self.relu(self.cmt_layer2(out_2))
        out_2 = torch.cat((out_0, out_2), dim=-3)

        out_3 = self.relu(self.conv_layer3(out_2))
        out_3 = self.relu(self.cmt_layer3(out_3))
        out_3 = torch.cat((out_0, out_3), dim=-3)

        out = self.head(out_3)
        out = self.relu(x + out)

        out = torch.clamp(out, min=0.0, max=1.0)
        return out

class MAP_ECMT_Generator(nn.Module):
    def __init__(self, depth=5):
        super(MAP_ECMT_Generator, self).__init__()        

        self.ECMT_2D = Light_ECMT_2D()
        self.depth   = depth

    def inference(self, x):
        self.ECMT_2D.eval()
        with torch.no_grad():
            for _ in range(self.depth):
                x = self.ECMT_2D(x)
        return x
        
    def forward(self, x):
        results = []
        for _ in range(self.depth):
            x = self.ECMT_2D(x)
            results.append(x)
        return results

class MAP_ECMT(nn.Module):
    def __init__(self):
        super(MAP_ECMT, self).__init__()
        self.Generator         = MAP_ECMT_Generator()
        
        self.Discriminator     = turn_on_spectral_norm( UNet(repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False) )
        
        self.gan_metric        = ls_gan
        self.mse_criterion     = nn.MSELoss(reduction='none')
        self.sobel             = SobelOperator()

    def d_loss(self, x, y):
        fake  = self.Generator(x)
        fake  = fake[-1]
        fake_enc, fake_dec     = self.Discriminator(fake.detach())
        real_enc, real_dec     = self.Discriminator(y)
        source_enc, source_dec = self.Discriminator(x)
        
        d_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + \
                 self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.) + \
                 self.gan_metric(source_enc, 0.) + self.gan_metric(source_dec, 0.)

        return d_loss


    def g_loss(self, x, y):
        fake               = self.Generator(x)
        fake_enc, fake_dec = self.Discriminator(fake[-1])
        gan_loss           = self.gan_metric(fake_enc, 1.) + self.gan_metric(fake_dec, 1.)

        mse_loss1  = torch.mean(self.mse_criterion(fake[-4], y[-4]))
        mse_loss2  = torch.mean(self.mse_criterion(fake[-3], y[-3]))
        mse_loss3  = torch.mean(self.mse_criterion(fake[-2], y[-2]))
        mse_loss4  = torch.mean(self.mse_criterion(fake[-1], y[-1]))
        
        mse_loss   = (mse_loss1 + mse_loss2 + mse_loss3 + mse_loss4) / 4.0

        # custom
        # self.sobel = self.sobel.to('cpu')
        # weight_map = ROI_map_extract(y[-1].clone().cpu().detach()) / self.sobel(y[-1].clone().cpu().detach())
        # weight_map = weight_map.float().to('cuda')
        
        # focus_loss = torch.mean( self.mse_criterion(fake[-1], y[-1]) * weight_map )
 
        g_loss = gan_loss + 50.0*mse_loss #+ 50.0*focus_loss
        # return (g_loss, gan_loss, mse_loss, focus_loss)
        return (g_loss, gan_loss, mse_loss)




# 2. Main Unet GAN Model
class Unet_GAN(nn.Module):
    def __init__(self):
        super(Unet_GAN, self).__init__()
        self.Generator         = Revised_UNet()
        self.Discriminator     = turn_on_spectral_norm( UNet(repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False) )
        
        self.gan_metric        = ls_gan
        self.criterion         = Perceptual_L1_Loss()

    def inference(self, x):
        with torch.no_grad():
            fake = self.Generator(x)
        return fake

    def d_loss(self, x, y):
        fake  = self.Generator(x)

        fake_enc, fake_dec     = self.Discriminator(fake.detach())
        real_enc, real_dec     = self.Discriminator(y)
        source_enc, source_dec = self.Discriminator(x)
        
        d_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + \
                 self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.) + \
                 self.gan_metric(source_enc, 0.) + self.gan_metric(source_dec, 0.)

        return d_loss


    def g_loss(self, x, y):
        fake               = self.Generator(x)
        fake_enc, fake_dec = self.Discriminator(fake)

        gan_loss           = self.gan_metric(fake_enc, 1.) + self.gan_metric(fake_dec, 1.)
        pix_loss           = self.criterion(fake, y)

        g_loss = gan_loss + 50.0*pix_loss #+ 50.0*focus_loss

        return (g_loss, gan_loss, pix_loss)




# 3. Main Mixed Unet_GAN Model - RRDB_Net Version
class Mixed_Unet_GAN_A(nn.Module):
    def __init__(self):
        super(Mixed_Unet_GAN_A, self).__init__()
        self.feat_extractor    = Revised_UNet()
        self.Generator         = RRDB_Net(in_nc=1, out_nc=1, nf=64, nb=23, gc=32)
        self.Discriminator     = turn_on_spectral_norm( UNet(repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False) )
        
        self.gan_metric        = ls_gan
        self.criterion         = Perceptual_L1_Loss()

        # pre-trained feat extractor
        print("Load feature extractor...!")
        checkpoint = torch.load("/workspace/sunggu/4.Dose_img2img/model/[Ours]Revised_UNet/epoch_991_checkpoint.pth", map_location='cpu')
        self.feat_extractor.load_state_dict(checkpoint['model_state_dict'])
        for p in self.feat_extractor.parameters():
            p.requires_grad = False

    def inference(self, x):
        with torch.no_grad():
            fake = self.Generator(self.feat_extractor(x))

        return fake

    def d_loss(self, x, y):
        fake  = self.Generator(self.feat_extractor(x))

        fake_enc, fake_dec     = self.Discriminator(fake.detach())
        real_enc, real_dec     = self.Discriminator(y)
        source_enc, source_dec = self.Discriminator(x)
        
        d_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + \
                 self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.) + \
                 self.gan_metric(source_enc, 0.) + self.gan_metric(source_dec, 0.)

        return d_loss


    def g_loss(self, x, y):
        fake               = self.Generator(self.feat_extractor(x))
        fake_enc, fake_dec = self.Discriminator(fake)
        
        gan_loss           = self.gan_metric(fake_enc, 1.) + self.gan_metric(fake_dec, 1.)
        pix_loss           = self.criterion(fake, y)

        g_loss = gan_loss + 50.0*pix_loss

        return (g_loss, gan_loss, pix_loss)




# 4. WDCNN Version
def normalize_gradient_enc_dec(net_D, x, **kwargs):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |

    reference : https://github.com/basiclab/GNGAN-PyTorch
    """
    x.requires_grad_(True)
    f_enc, f_dec  = net_D(x, **kwargs)

    # encoder
    enc_grad      = torch.autograd.grad(f_enc, [x], torch.ones_like(f_enc), create_graph=True, retain_graph=True)[0]
    enc_grad_norm = torch.norm(torch.flatten(enc_grad, start_dim=1), p=2, dim=1)
    enc_grad_norm = enc_grad_norm.view(-1, *[1 for _ in range(len(f_enc.shape) - 1)])
    enc_f_hat     = (f_enc / (enc_grad_norm + torch.abs(f_enc)))

    # decoder
    dec_grad      = torch.autograd.grad(f_dec, [x], torch.ones_like(f_dec), create_graph=True, retain_graph=True)[0]
    dec_grad_norm = torch.norm(torch.flatten(dec_grad, start_dim=1), p=2, dim=1)
    dec_grad_norm = dec_grad_norm.view(-1, *[1 for _ in range(len(f_dec.shape) - 1)])
    dec_f_hat     = (f_dec / (dec_grad_norm + torch.abs(f_dec)))

    return enc_f_hat, dec_f_hat

class Mixed_Unet_GAN_B(nn.Module):
    def __init__(self):
        super(Mixed_Unet_GAN_B, self).__init__()
        self.feat_extractor    = Revised_UNet()
        self.Generator         = WDCNN()
        self.Discriminator     = UNet(repeat_num=6, use_discriminator=True, conv_dim=64, use_sigmoid=False)
        
        self.gan_metric        = nn.BCEWithLogitsLoss()
        self.criterion         = Perceptual_L1_Loss()

        # pre-trained feat extractor
        print("Load feature extractor...!")
        checkpoint = torch.load("/workspace/sunggu/4.Dose_img2img/model/[Ours]Revised_UNet/epoch_991_checkpoint.pth", map_location='cpu')
        self.feat_extractor.load_state_dict(checkpoint['model_state_dict'])
        for p in self.feat_extractor.parameters():
            p.requires_grad = False

    def inference(self, x):
        with torch.no_grad():
            fake = self.Generator(self.feat_extractor(x))

        return fake

    # Add Gradient Norm
    def d_loss(self, x, y):
        fake  = self.Generator(self.feat_extractor(x))

        fake_enc, fake_dec     = normalize_gradient_enc_dec(self.Discriminator, fake.detach())
        real_enc, real_dec     = normalize_gradient_enc_dec(self.Discriminator, y)
    
        source_enc, source_dec = normalize_gradient_enc_dec(self.Discriminator, x)
        
        d_loss = self.gan_metric(real_enc, torch.ones_like(real_enc)) + self.gan_metric(real_dec, torch.ones_like(real_dec)) + \
                 self.gan_metric(fake_enc, torch.zeros_like(fake_enc)) + self.gan_metric(fake_dec, torch.zeros_like(fake_dec)) + \
                 self.gan_metric(source_enc, torch.zeros_like(source_enc)) + self.gan_metric(source_dec, torch.zeros_like(source_dec))

        return d_loss


    def g_loss(self, x, y):
        fake               = self.Generator(self.feat_extractor(x))
        fake_enc, fake_dec = self.Discriminator(fake)
        
        gan_loss           = self.gan_metric(fake_enc, torch.ones_like(fake_enc)) + self.gan_metric(fake_dec, torch.ones_like(fake_dec))
        pix_loss           = self.criterion(fake, y)

        g_loss = gan_loss + 50.0*pix_loss

        return (g_loss, gan_loss, pix_loss)



# 5. GPEN_GAN Version - ref : https://github.com/basiclab/gngan-pytorch
def normalize_gradient_enc(net_D, x, **kwargs):
    x.requires_grad_(True)
    # f_enc  = net_D(x, **kwargs)
    f_enc  = net_D(DiffAugment(x, policy='color,translation,cutout'), **kwargs)
    
    # encoder
    enc_grad      = torch.autograd.grad(f_enc, [x], torch.ones_like(f_enc), create_graph=True, retain_graph=True)[0]
    enc_grad_norm = torch.norm(torch.flatten(enc_grad, start_dim=1), p=2, dim=1)
    enc_grad_norm = enc_grad_norm.view(-1, *[1 for _ in range(len(f_enc.shape) - 1)])
    enc_f_hat     = (f_enc / (enc_grad_norm + torch.abs(f_enc)))

    return enc_f_hat

class GPEN_GAN(nn.Module):
    def __init__(self):
        super(GPEN_GAN, self).__init__()
        self.Generator         = GPEN_FullGenerator(size=64, style_dim=512, n_mlp=8, channel_multiplier=2, narrow=1.0, device='cuda')    
        self.Discriminator     = GPEN_Discriminator(size=64, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], narrow=1.0, device='cuda')

        self.gan_metric        = nn.BCEWithLogitsLoss()
        self.criterion         = Perceptual_L1_Loss()


    def inference(self, x):
        with torch.no_grad():
            fake = self.Generator(x)

        return fake

    # Add Gradient Norm
    def d_loss(self, x, y):
        fake  = self.Generator(x)

        fake_enc     = normalize_gradient_enc(self.Discriminator, fake.detach())
        real_enc     = normalize_gradient_enc(self.Discriminator, y)
    
        source_enc   = normalize_gradient_enc(self.Discriminator, x)
        
        d_loss = self.gan_metric(real_enc, torch.ones_like(real_enc)) + \
                 self.gan_metric(fake_enc, torch.zeros_like(fake_enc)) + \
                 self.gan_metric(source_enc, torch.zeros_like(source_enc))

        return d_loss


    def g_loss(self, x, y):
        fake               = self.Generator(x)
        fake_enc           = self.Discriminator(fake)
        
        gan_loss           = self.gan_metric(fake_enc, torch.ones_like(fake_enc))
        pix_loss           = self.criterion(fake, y)

        g_loss = gan_loss + 10.0*pix_loss

        return (g_loss, gan_loss, pix_loss)




# 6. Revised Version
class Revised_GPEN_GAN(nn.Module):
    def __init__(self):
        super(Revised_GPEN_GAN, self).__init__()
        self.Generator         = Ours_GPEN_FullGenerator(size=64, style_dim=512, n_mlp=8, channel_multiplier=2, narrow=1.0, device='cuda')    
        self.Discriminator     = Ours_GPEN_Discriminator(size=64, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], narrow=1.0, device='cuda')

        self.gan_metric        = nn.BCEWithLogitsLoss()
        self.criterion         = Perceptual_L1_Loss()


    def inference(self, x):
        with torch.no_grad():
            fake = self.Generator(x)

        return fake

    # Add Gradient Norm
    def d_loss(self, x, y):
        fake  = self.Generator(x)

        fake_enc     = normalize_gradient_enc(self.Discriminator, fake.detach())
        real_enc     = normalize_gradient_enc(self.Discriminator, y)
        source_enc   = normalize_gradient_enc(self.Discriminator, x)
        
        d_loss = self.gan_metric(real_enc, torch.ones_like(real_enc)) + \
                 self.gan_metric(fake_enc, torch.zeros_like(fake_enc)) + \
                 self.gan_metric(source_enc, torch.zeros_like(source_enc))

        return d_loss


    def g_loss(self, x, y):
        fake               = self.Generator(x)
        # fake_enc           = self.Discriminator(fake)
        fake_enc           = self.Discriminator(DiffAugment(fake, policy='color,translation,cutout'))
        
        gan_loss           = self.gan_metric(fake_enc, torch.ones_like(fake_enc))
        pix_loss           = self.criterion(fake, y)

        g_loss = gan_loss + 10.0*pix_loss

        return (g_loss, gan_loss, pix_loss)







