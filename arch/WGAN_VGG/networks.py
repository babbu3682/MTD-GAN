import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19


class WGAN_VGG_Generator(nn.Module):
    def __init__(self):
        super(WGAN_VGG_Generator, self).__init__()
        layers = [nn.Conv2d(1,32,3,1,1), nn.ReLU()]
        for i in range(2, 8):
            layers.append(nn.Conv2d(32,32,3,1,1))
            layers.append(nn.ReLU())
            
        layers.extend([nn.Conv2d(32,1,3,1,1), nn.ReLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class WGAN_VGG_Discriminator(nn.Module):
    def __init__(self, input_size):
        super(WGAN_VGG_Discriminator, self).__init__()
        
        def conv_output_size(input_size, kernel_size_list, stride_list):
            n = (input_size - kernel_size_list[0]) // stride_list[0] + 1
            for k, s in zip(kernel_size_list[1:], stride_list[1:]):
                n = (n - k) // s + 1
            return n

        def add_block(layers, ch_in, ch_out, stride):
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            layers.append(nn.LeakyReLU())
            return layers

        layers = []
        ch_stride_set = [(1,64,1),(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2)]
        for ch_in, ch_out, stride in ch_stride_set:
            add_block(layers, ch_in, ch_out, stride)

        self.output_size = conv_output_size(input_size, [3]*6, [1,2]*3)
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256*self.output_size*self.output_size, 1024)
        self.lrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, 1)
        

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out


#  Reference: https://github.com/SSinyu/WGAN-VGG/blob/master/networks.py
#  Reference: https://github.com/yyqqss09/ldct_denoising/blob/master/models.py
#  But it has a few error lines.

# class WGAN_VGG_Generator(nn.Module):
#     def __init__(self):
#         super(WGAN_VGG_Generator, self).__init__()
#         self.up1 = nn.Upsample(size=80, mode='bicubic')
#         self.up2 = nn.Upsample(size=528, mode='bicubic')
#         layers = [nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False), nn.ReLU()]
#         for i in range(2, 8):
#             layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False))
#             layers.append(nn.ReLU())
            
#         layers.extend([nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False), nn.ReLU()])
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         if self.training:
#             x   = self.up1(x)
#         else :
#             x   = self.up2(x)
#         out = self.net(x)
#         return out


# class WGAN_VGG_Discriminator(nn.Module):
#     def __init__(self):
#         super(WGAN_VGG_Discriminator, self).__init__()
        
#         def add_block(layers, ch_in, ch_out, stride):
#             layers.append(nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=stride, padding=1))
#             layers.append(nn.LeakyReLU(0.2))
#             return layers

#         layers = []
#         ch_stride_set = [(1,64,1),(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2)]
#         for ch_in, ch_out, stride in ch_stride_set:
#             add_block(layers, ch_in, ch_out, stride)

#         self.net   = nn.Sequential(*layers)
#         self.fc1   = nn.Linear(256*8*8, 1024)
#         self.lrelu = nn.LeakyReLU(0.2)
#         self.fc2   = nn.Linear(1024, 1)
        

#     def forward(self, x):
#         out = self.net(x)
#         out = out.flatten(1)
#         out = self.fc1(out)
#         out = self.lrelu(out)
#         out = self.fc2(out)
#         return out

# class WGAN_VGG_Generator(nn.Module):
#     def __init__(self):
#         super(WGAN_VGG_Generator, self).__init__()
#         layers = [nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU()]
#         for i in range(2, 8):
#             layers.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False))
#             layers.append(nn.ReLU())
            
#         layers.extend([nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU()])
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.net(x)
#         return out


# class WGAN_VGG_Discriminator(nn.Module):
#     def __init__(self):
#         super(WGAN_VGG_Discriminator, self).__init__()
        
#         def add_block(layers, ch_in, ch_out, stride):
#             layers.append(nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=stride, padding=1))
#             layers.append(nn.LeakyReLU(0.2))
#             return layers

#         layers = []
#         ch_stride_set = [(1,64,1),(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2)]
#         for ch_in, ch_out, stride in ch_stride_set:
#             add_block(layers, ch_in, ch_out, stride)

#         self.net   = nn.Sequential(*layers)
#         self.fc1   = nn.Linear(256*8*8, 1024)
#         self.lrelu = nn.LeakyReLU(0.2)
#         self.fc2   = nn.Linear(1024, 1)
        

#     def forward(self, x):
#         out = self.net(x)
#         out = out.flatten(1)
#         out = self.fc1(out)
#         out = self.lrelu(out)
#         out = self.fc2(out)
#         return out

class WGAN_VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(WGAN_VGG_FeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(*list(vgg19(pretrained=True).features.children())[:35])
        # for fixed feat extractor
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.feature_extractor(x)
        return out


class WGAN_VGG(nn.Module):
    def __init__(self, input_size=64):
        super(WGAN_VGG, self).__init__()
        self.Generator         = WGAN_VGG_Generator()
        self.Discriminator     = WGAN_VGG_Discriminator(input_size)
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion       = nn.MSELoss()

    # # referred from https://github.com/kuc2477/pytorch-wgan-gp
    # def __init__(self):
    #     super(WGAN_VGG, self).__init__()
    #     self.Generator         = WGAN_VGG_Generator()
    #     self.Discriminator     = WGAN_VGG_Discriminator()
    #     self.feature_extractor = WGAN_VGG_FeatureExtractor()
    #     self.p_criterion       = nn.MSELoss()        

    def d_loss(self, x, y, gp=True, return_gp=False):
        fake   = self.Generator(x).detach()   # fixed this line.
        d_fake = self.Discriminator(fake)
        d_real = self.Discriminator(y)
        
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)

        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss

        return (loss, gp_loss) if return_gp else loss

    def g_loss(self, x, y, perceptual=True, return_p=False):
        fake   = self.Generator(x)
        d_fake = self.Discriminator(fake)

        g_loss = -torch.mean(d_fake)

        if perceptual:
            p_loss = self.p_loss(fake=fake, real=y)
            loss = g_loss + (0.1 * p_loss)
        else:
            p_loss = None
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    def p_loss(self, fake, real):
        fake_feature = self.feature_extractor(fake.repeat(1,3,1,1))
        real_feature = self.feature_extractor(real.repeat(1,3,1,1))
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10.0):
        assert y.size() == fake.size()
        a = torch.cuda.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.Discriminator(interp)
        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(outputs=d_interp, inputs=interp, grad_outputs=fake_, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty
