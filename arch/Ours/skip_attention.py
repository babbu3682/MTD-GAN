import torch
import torch.nn as nn

class SkipAttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(SkipAttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(F_int)
            nn.InstanceNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(F_int)
            nn.InstanceNorm2d(F_int)
        )
        
        # psi = 프사이(수학기호)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(1),
            nn.InstanceNorm2d(1),       
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, skip):
        # x = skip connection feature, g = down feature

        g_feat          = self.W_g(g)
        skip_feat       = self.W_x(skip)
        mixed_feat      = self.relu(g_feat+skip_feat)
        atten_map       = self.psi(mixed_feat)
        return atten_map * skip



class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)        