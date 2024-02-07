import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
import torch.nn.functional as F


# Revised Unet
class DownsampleBlock(nn.Module):
    def __init__(self, scale, input_channels):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=scale),
            nn.Conv2d(input_channels*scale**2, input_channels, kernel_size=1, stride=1, padding=1//2),
            nn.PReLU()
        )

    def forward(self, input):
        return self.downsample(input)

class UpsampleBlock(nn.Module):
    def __init__(self, scale, input_channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, input_channels*scale**2, kernel_size=1, stride=1, padding=1//2),
            nn.PixelShuffle(upscale_factor=scale),
            nn.PReLU()
        )

    def forward(self, input):
        return self.upsample(input)

class Revised_UNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(Revised_UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=input_nc, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool1 = DownsampleBlock(scale=2, input_channels=64)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = DownsampleBlock(scale=2, input_channels=128)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = DownsampleBlock(scale=2, input_channels=256)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        # self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = DownsampleBlock(scale=2, input_channels=512)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)
        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        # self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool4 = UpsampleBlock(scale=2, input_channels=512)
        
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        # self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool3 = UpsampleBlock(scale=2, input_channels=256)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        # self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool2 = UpsampleBlock(scale=2, input_channels=128)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        # self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool1 = UpsampleBlock(scale=2, input_channels=64)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc   = nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        output = self.fc(dec1_1)
        output = self.relu(output + x)

        return output




#####################################################################################################################
######################################       MIXER Factory     ######################################################
#####################################################################################################################

# MLP Mixer
# Original MLPMixer Paper: https://arxiv.org/pdf/2105.01601.pdf
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, int(dim*expansion_factor)),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(int(dim*expansion_factor), dim),
        nn.Dropout(dropout)
    )

class MLPMixer(nn.Module):
    def __init__(self, image_size, channels, patch_size, dim, depth, expansion_factor = (0.5, 4), dropout = 0.):
        super(MLPMixer, self).__init__()
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        
        self.denoiser = Revised_UNet()

        self.patch_embed = nn.Sequential( Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size), 
                                          nn.Linear((patch_size ** 2) * channels, dim) )

        self.mixer_blocks = nn.Sequential(*[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor[0], dropout, chan_first)),  # token mixing
            PreNormResidual(dim, FeedForward(dim, expansion_factor[1], dropout, chan_last))            # channel mixing
            ) for _ in range(depth)])

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_size*patch_size*channels),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = int(image_size/patch_size), p1 = patch_size, p2 = patch_size),
            )

        # pre-trained feat extractor
        print("Load feature extractor...!")
        checkpoint = torch.load("/workspace/sunggu/4.Dose_img2img/model/[Ours]Revised_UNet/epoch_991_checkpoint.pth", map_location='cpu')
        self.denoiser.load_state_dict(checkpoint['model_state_dict'])
        for p in self.denoiser.parameters():
            p.requires_grad = False


    def forward(self, input):
        with torch.no_grad():
            input = self.denoiser(input)

        x = self.patch_embed(input)    
        x = self.mixer_blocks(x)    
        x = self.head(x)    

        output = input + x

        return torch.nn.functional.relu(output)





# Img2Img Mixer
# MLPMixer img2img translation Paper: https://openreview.net/pdf?id=wsuQ2h6KZXQ
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b h w c")
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    
class PatchExpansion(nn.Module):
    def __init__(self, dim_scale, channel_dim, img_channels, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_scale = dim_scale
        self.expand = nn.Linear(channel_dim, dim_scale**2* channel_dim, bias=False)
        self.output_dim = channel_dim 
        self.norm = norm_layer(channel_dim)
        self.output = nn.Conv2d(in_channels=channel_dim,out_channels=img_channels ,kernel_size=1,bias=False)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.expand(x)
        B, H, W, C = x.shape

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        
        x = x.view(B, H*self.dim_scale, W*self.dim_scale, -1)
        x = x.permute(0,3,1,2)
        x = self.output(x)

        return x
    
class MLPBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int,  dropout = 0.):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Mixer(nn.Module):

    def __init__(self, num_patches: int, num_channels: int, f_hidden: int):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            Rearrange("b h w c -> b c w h"),
            MLPBlock(num_patches, num_patches*f_hidden),
            Rearrange("b c w h -> b c h w"),
            MLPBlock(num_patches, num_patches*f_hidden),
            Rearrange("b c h w -> b h w c"),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(num_channels),
            MLPBlock(num_channels, num_channels*f_hidden)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x

class Img2Img_Mixer(nn.Module):

    def __init__(self, img_size: int = 512, img_channels: int = 1, patch_size: int = 4, embed_dim: int = 128, num_layers: int = 16, f_hidden: int = 8):    
        super().__init__()
        
        self.patch_embed = PatchEmbedding(patch_size, embed_dim, img_channels)
    
        self.mixer_layers = nn.Sequential(*[ Mixer(img_size//patch_size, embed_dim, f_hidden) for _ in range(num_layers)])

        self.patch_expand = PatchExpansion(patch_size, embed_dim, img_channels)
        
        self.denoiser = Revised_UNet()


        # pre-trained feat extractor
        print("Load feature extractor...!")
        checkpoint = torch.load("/workspace/sunggu/4.Dose_img2img/model/[Ours]Revised_UNet/epoch_991_checkpoint.pth", map_location='cpu')
        self.denoiser.load_state_dict(checkpoint['model_state_dict'])
        for p in self.denoiser.parameters():
            p.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            input = self.denoiser(input)

        x = self.patch_embed(input)
        x = self.mixer_layers(x)
        x = self.patch_expand(x)
        
        output = input + x

        return torch.nn.functional.relu(output)






# High frequency ConvMixer
# ConvMixer Paper: https://arxiv.org/pdf/2201.09792.pdf
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

        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, group=1, padding=0):
        super(ConvMixer_Block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, groups=group, padding=padding)
        self.gelu = nn.GELU()
        self.norm = nn.BatchNorm2d(out_ch)

        # Initialize by xavier_uniform_
        self.init_weight()
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        x = self.norm(x)
        return x

class HF_ConvMixer(nn.Module):
    def __init__(self, kernel_size=9, patch_size=4, dim=1536, hf_ch=20):
        super(HF_ConvMixer, self).__init__()

        self.hf_conv     = HF_Conv(in_channels=1, out_channels=hf_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.patch_embed = ConvMixer_Block(in_ch=20+1, out_ch=dim//2, kernel_size=patch_size, stride=patch_size)
        
        # depth = 8
        self.mixer_block1 = nn.Sequential( Residual(ConvMixer_Block(in_ch=dim//2, out_ch=dim//2, kernel_size=kernel_size, group=dim//2, padding="same")), ConvMixer_Block(in_ch=dim//2, out_ch=dim//2, kernel_size=1) )
        self.mixer_block2 = nn.Sequential( Residual(ConvMixer_Block(in_ch=dim, out_ch=dim, kernel_size=kernel_size, group=dim, padding="same")), ConvMixer_Block(in_ch=dim, out_ch=dim//2, kernel_size=1) )
        self.mixer_block3 = nn.Sequential( Residual(ConvMixer_Block(in_ch=dim, out_ch=dim, kernel_size=kernel_size, group=dim, padding="same")), ConvMixer_Block(in_ch=dim, out_ch=dim//2, kernel_size=1) )
        self.mixer_block4 = nn.Sequential( Residual(ConvMixer_Block(in_ch=dim, out_ch=dim, kernel_size=kernel_size, group=dim, padding="same")), ConvMixer_Block(in_ch=dim, out_ch=dim//2, kernel_size=1) )
        self.mixer_block5 = nn.Sequential( Residual(ConvMixer_Block(in_ch=dim, out_ch=dim, kernel_size=kernel_size, group=dim, padding="same")), ConvMixer_Block(in_ch=dim, out_ch=dim//2, kernel_size=1) )
        self.mixer_block6 = nn.Sequential( Residual(ConvMixer_Block(in_ch=dim, out_ch=dim, kernel_size=kernel_size, group=dim, padding="same")), ConvMixer_Block(in_ch=dim, out_ch=dim//2, kernel_size=1) )
        self.mixer_block7 = nn.Sequential( Residual(ConvMixer_Block(in_ch=dim, out_ch=dim, kernel_size=kernel_size, group=dim, padding="same")), ConvMixer_Block(in_ch=dim, out_ch=dim//2, kernel_size=1) )
        self.mixer_block8 = nn.Sequential( Residual(ConvMixer_Block(in_ch=dim, out_ch=dim, kernel_size=kernel_size, group=dim, padding="same")), ConvMixer_Block(in_ch=dim, out_ch=dim//2, kernel_size=1) )

        self.upsample = nn.Sequential( nn.Conv2d(dim//2, (20+1)*patch_size**2, kernel_size=1, stride=1, padding=0), nn.PixelShuffle(upscale_factor=patch_size), nn.PReLU() )
        self.head     = nn.Conv2d(in_channels=20+1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu     = nn.ReLU()

    def forward(self, input):
        x = self.hf_conv(input)     
        x = torch.cat((x, input), dim=1)  # x  == (B, 21, 512, 512)
        
        x0 = self.patch_embed(x)          # x0 == [1, 768, 128, 128]
        
        x1 = self.mixer_block1(x0)        # x1 == [1, 768, 128, 128]
        x1 = torch.cat((x1, x0), dim=1)   # x1 == (B, 768*2, 128, 128)   

        x2 = self.mixer_block2(x1)        # x2 == (B, 768*2, 128, 128)   
        x2 = torch.cat((x2, x0), dim=1)   # x2 == (B, 768*2, 128, 128)   
        
        x3 = self.mixer_block3(x2)  
        x3 = torch.cat((x3, x0), dim=1)
        
        x4 = self.mixer_block4(x3)  
        x4 = torch.cat((x4, x0), dim=1)

        x5 = self.mixer_block5(x4)  
        x5 = torch.cat((x5, x0), dim=1)

        x6 = self.mixer_block6(x5)  
        x6 = torch.cat((x6, x0), dim=1)

        x7 = self.mixer_block7(x6)  
        x7 = torch.cat((x7, x0), dim=1)

        x8 = self.mixer_block8(x7)        # x8 == (B, 768, 128, 128)   

        x8 = self.upsample(x8)            # x8 == [1, 21, 512, 512]
        x8 = self.head(x8)                # x8 == [1, 1, 512, 512]
        output  = self.relu(x8+input)    

        return output

