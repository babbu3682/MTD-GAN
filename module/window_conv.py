import torch
import torch.nn as nn
import torch.nn.functional as F

# reference : https://github.com/MGH-LMIC/windows_optimization (Practical Window Setting Optimization for Medical Image Deep Learning: https://arxiv.org/abs/1812.00572)

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
