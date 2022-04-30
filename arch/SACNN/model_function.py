'''
@Description: 
@Author: GuoYi
@Date: 2020-06-15 10:04:32
@LastEditTime: 2020-06-28 17:43:02
@LastEditors: GuoYi
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reference : https://github.com/mmlipku/sacnn/blob/main/sacnn.py
## Self-Attention Block
##***********************************************************************************************************
class SA_Block(nn.Module):
    """
    input:N*C*D*H*W
    """
    def __init__(self, in_ch):
        super().__init__()
        # self.N = N 
        # self.C = in_ch
        # self.D = 3
        # self.H = 64
        # self.W = 64
        self.in_ch  = in_ch

        self.gamma = nn.Parameter(torch.zeros(1))

        self.conv3d_k1_a  = nn.Conv3d(in_channels=self.in_ch, out_channels=self.in_ch//8, kernel_size=1)           # key
        self.conv3d_k1_b  = nn.Conv3d(in_channels=self.in_ch, out_channels=self.in_ch//8, kernel_size=1)           # quary
        self.conv3d_k3    = nn.Conv3d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=3, padding=1)   # value

        self.softmax  = nn.Softmax(dim=-1)

    def Cal_P_att(self, k_x, q_x, v_x):
        B, C, D, H, W = v_x.shape

        k_x = k_x.reshape((B, -1, H*W)).transpose(1, 2)
        q_x = q_x.reshape((B, -1, H*W))

        cor = torch.bmm(k_x, q_x)    
        cor = self.softmax(cor).transpose(1, 2)

        v_x = v_x.reshape((B, C*D, H*W))
        Patt = torch.bmm(v_x, cor).reshape(B, C, D, H, W)
        return Patt

    def Cal_D_att(self, k_x, q_x, v_x):
        B, C, D, H, W = v_x.shape
        
        k_x = k_x.transpose(1, 2).reshape(B, D, -1)                   # (B, D,     C*H*W)
        q_x = q_x.transpose(1, 2).reshape(B, D, -1).transpose(1, 2)   # (B, C*H*W, D)
        
        cor = torch.bmm(k_x, q_x)                                  # (B, D,    D)
        cor = self.softmax(cor).transpose(1, 2)                    # (B, s(D), D)

        v_x = v_x.transpose(1, 2).reshape(B, D, -1).transpose(1, 2)   # v_x = (B, C*H*W, D)                
                                                                      # cor = (B, s(D),  D)                

        D_att = torch.bmm(v_x, cor).transpose(1, 2)             # (B, D, C*H*W)

        D_att = D_att.view(B, D, C, H, W).transpose(1, 2)          # (B, C, D, H, W)

        return D_att


    def forward(self, x):

        k_x = self.conv3d_k1_a(x)
        q_x = self.conv3d_k1_b(x)
        v_x = self.conv3d_k3(x)
        
        Patt = self.Cal_P_att(k_x, q_x, v_x)
        Datt = self.Cal_D_att(k_x, q_x, v_x)

        Y = self.gamma*(Patt + Datt) + x

        return Y



## 3D Convolutional
##***********************************************************************************************************
class Conv3D_Block(nn.Module):
    # input shape: N,C,D,H,W 

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Conv3d input:N*C*D*H*W
        # Conv3d output:N*C*D*H*W

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv3d(x)
        return out


######################################################################################################
########## AutoEncoder

class AE_Conv2D_Block(nn.Module):
    """
    input:N*C*D*H*W
    """
    def __init__(self, in_channels, out_channels):
        super(AE_Conv2D_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)



