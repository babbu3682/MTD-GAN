
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Reference : https://github.com/hmshan/MAP-NN/blob/f7a24be1981314b78f21d519ed330d6c7326d1f0/models.py
# But it was tensorflow version...


class CPCE_2D(nn.Module):
    def __init__(self):
        super(CPCE_2D, self).__init__()
        # Encoder
        self.encoder1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False)            
        self.encoder2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False)
        self.encoder3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False)
        self.encoder4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False)

        # Decoder
        self.decoder5  = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False)
        self.decoder5_ = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.decoder6  = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False)
        self.decoder6_ = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.decoder7  = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False)
        self.decoder7_ = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Head
        self.decoder8  = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

        # Initialize by xavier_uniform_
        # self.apply(self.init_weights)
        self.init_weights()

    def forward(self, x):
        # x shape is (B, C, H, W)

        # Encoder
        x1 = self.encoder1(x)
        x2 = F.relu(x1)

        x2 = self.encoder2(x2)
        x3 = F.relu(x2)

        x3 = self.encoder3(x3)
        x4 = F.relu(x3)

        x4 = self.encoder4(x4)
        x5 = F.relu(x4)

        # Decoder
        x5  = self.decoder5(x5)
        x5  = F.relu(torch.cat([x3, x5], dim=1))
        x6  = F.relu(self.decoder5_(x5))

        x6  = self.decoder6(x6)
        x6  = F.relu(torch.cat([x2, x6], dim=1))
        x7  = F.relu(self.decoder6_(x6))

        x7  = self.decoder7(x7)
        x7  = F.relu(torch.cat([x1, x7], dim=1))
        x8  = F.relu(self.decoder7_(x7))                  

        x8  = self.decoder8(x8)
        x   = F.relu(x+x8)

        x   = torch.clamp(x, min=0.0, max=1.0)

        return x

    def init_weights(self):
        print("inintializing...!")
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

'''
MAP-NN 
inputs: N x input_width x input_height x 1
'''
class MAP_NN_Generator(nn.Module):
    def __init__(self, depth=5):
        super(MAP_NN_Generator, self).__init__()        

        self.CPCE_2D = CPCE_2D()
        self.depth   = depth


    def forward(self, x):
        for _ in range(self.depth):
            x = self.CPCE_2D(x)
        return x
  

class MAP_NN_Discriminator(nn.Module):
    def __init__(self):
        super(MAP_NN_Discriminator, self).__init__()       
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)            
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)            

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)            
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True)            

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)            
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True)            

        self.fc1  = nn.Linear(in_features=16384, out_features=1024, bias=True)  # 8 x 8 x 256 = 16,384
        self.fc2  = nn.Linear(in_features=1024, out_features=1, bias=True)

    def forward(self, x):

        x = self.conv1(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv3(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv4(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv5(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv6(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        # print(x.flatten(start_dim=1, end_dim=-1).shape)
        x = self.fc1(x.flatten(start_dim=1, end_dim=-1))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.fc2(x)

        return x

class SobelOperator(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.register_buffer('conv_x', torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, None, :, :] / 4)
        self.register_buffer('conv_y', torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None, :, :] / 4)

    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)

        grad_x = F.conv2d(x, self.conv_x, bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x, self.conv_y, bias=None, stride=1, padding=1)

        x = torch.sqrt(grad_x**2 + grad_y**2 + self.epsilon)

        x = x.view(b, c, h, w)

        return x

class MAP_NN(nn.Module):
    # referred from https://github.com/kuc2477/pytorch-wgan-gp
    def __init__(self):
        super(MAP_NN, self).__init__()
        self.Generator         = MAP_NN_Generator()
        self.Discriminator     = MAP_NN_Discriminator()
        
        self.mse_criterion     = nn.MSELoss()
        self.sobel             = SobelOperator()

    def gp(self, y, fake, lambda_=10):
        assert y.size() == fake.size()
        a = torch.cuda.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.Discriminator(interp)
        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(outputs=d_interp, inputs=interp, grad_outputs=fake_, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

    def d_loss(self, x, y, gp=True, return_gp=False):
        fake   = self.Generator(x).detach()
        d_real = self.Discriminator(y)
        d_fake = self.Discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss     # MAP-NN paper use 10 scalar for GP
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss


    def g_loss(self, x, y):
        fake   = self.Generator(x)
        d_fake = self.Discriminator(fake)
        
        adv_loss  = -torch.mean(d_fake)
        mse_loss  = torch.mean(self.mse_criterion(fake, y))
        edge_loss = torch.mean(self.mse_criterion(self.sobel(fake), self.sobel(y)))
 
        g_loss = adv_loss + 50.0*mse_loss + 50.0*edge_loss
        return (g_loss, adv_loss, mse_loss, edge_loss)



