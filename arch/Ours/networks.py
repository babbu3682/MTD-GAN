import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator
from itertools import chain
from losses import NDS_Loss, EdgeLoss, CharbonnierLoss, NDS_Loss, ls_gan

############################################################################################################
# MTD-GAN
############################################################################################################


# FFT-Conv Block (Ref: https://github.com/pkumivision/FFC/issues/3, https://github.com/INVOKERer/DeepRFT/blob/06f109e1c959cc7b3ea855d5d5feecf4d88f09a8/layers.py#L92)

class FFT_ConvBlock(nn.Module):
    def __init__(self, out_channels):
        super(FFT_ConvBlock, self).__init__()
        self.img_conv  = nn.Conv2d(out_channels,   out_channels,   kernel_size=3, stride=1, padding=1)
        self.fft_conv  = nn.Conv2d(out_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Fourier domain   
        _, _, H, W = x.shape
        fft = torch.fft.rfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        fft = torch.cat([fft.real, fft.imag], dim=1)
        fft = F.relu(self.fft_conv(fft))
        fft_real, fft_imag = torch.chunk(fft, 2, dim=1)        
        fft = torch.complex(fft_real, fft_imag)
        fft = torch.fft.irfft2(fft, s=(H, W), dim=(2, 3), norm='ortho')
        
        # Image domain  
        img = F.relu(self.img_conv(x))

        # Mixing (residual, image, fourier)
        output = x + img + fft
        return output        

class ResFFT_Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=96, num_layers=10, kernel_size=5, padding=0):
        super(ResFFT_Generator, self).__init__()
        encoder = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        decoder = [nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        enforce = []
        for _ in range(num_layers):
            encoder.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            decoder.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        for _ in range(21):
            enforce.append(FFT_ConvBlock(out_channels))
        self.enforce = nn.ModuleList(enforce)

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.encoder[0].parameters(),
            self.encoder[1].parameters(),
            self.encoder[2].parameters(),
            self.encoder[3].parameters(),
            self.encoder[4].parameters(),
            self.encoder[5].parameters(),
            self.encoder[6].parameters(),
            self.encoder[7].parameters(),
            self.encoder[8].parameters(),
            self.encoder[9].parameters(),
            self.encoder[10].parameters(),
            self.decoder[-1].parameters(),
            self.decoder[-2].parameters(),
            self.decoder[-3].parameters(),
            self.decoder[-4].parameters(),
            self.decoder[-5].parameters(),
            self.decoder[-6].parameters(),
            self.decoder[-7].parameters(),
            self.decoder[-8].parameters(),
            self.decoder[-9].parameters(),
            self.decoder[-10].parameters(),
            self.decoder[-11].parameters()
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return None

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.decoder[-11].parameters()

    def forward(self, x: torch.Tensor):
    
        e1 = F.relu(self.encoder[0](x))
        e1 = self.enforce[0](e1)

        e2 = F.relu(self.encoder[1](e1))
        e2 = self.enforce[1](e2)

        e3 = F.relu(self.encoder[2](e2))
        e3 = self.enforce[2](e3)

        e4 = F.relu(self.encoder[3](e3))
        e4 = self.enforce[3](e4)

        e5 = F.relu(self.encoder[4](e4))
        e5 = self.enforce[4](e5)

        e6 = F.relu(self.encoder[5](e5))
        e6 = self.enforce[5](e6)

        e7 = F.relu(self.encoder[6](e6))
        e7 = self.enforce[6](e7)

        e8 = F.relu(self.encoder[7](e7))
        e8 = self.enforce[7](e8)

        e9 = F.relu(self.encoder[8](e8))
        e9 = self.enforce[8](e9)

        e10 = F.relu(self.encoder[9](e9))
        e10 = self.enforce[9](e10)

        # Bottleneck
        x_b = F.relu(self.encoder[10](e10))
        x_b = self.enforce[10](x_b)                

        # Decoder
        d10 = F.relu(self.decoder[-1](x_b) + e10)

        d9  = self.enforce[11](d10)
        d9  = F.relu(self.decoder[-2](d9) + e9)

        d8  = self.enforce[12](d9)
        d8  = F.relu(self.decoder[-3](d8) + e8)

        d7  = self.enforce[13](d8)
        d7  = F.relu(self.decoder[-4](d7) + e7)
          
        d6  = self.enforce[14](d7)
        d6  = F.relu(self.decoder[-5](d6) + e6)

        d5  = self.enforce[15](d6)
        d5  = F.relu(self.decoder[-6](d5) + e5)

        d4  = self.enforce[16](d5)
        d4  = F.relu(self.decoder[-7](d4) + e4)

        d3  = self.enforce[17](d4)
        d3  = F.relu(self.decoder[-8](d3) + e3)

        d2  = self.enforce[18](d3)
        d2  = F.relu(self.decoder[-9](d2) + e2)

        d1  = self.enforce[19](d2)
        d1  = F.relu(self.decoder[-10](d1) + e1)

        d0  = self.enforce[20](d1)
        d0  = F.relu(self.decoder[-11](d0) + x)
        
        return d0

class UpsampleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels*(scale**2), kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(upscale_factor=scale)
        )

    def forward(self, input):
        return self.upsample(input)

class Multi_Task_Discriminator_Skip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv1    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu1    = nn.LeakyReLU(0.2)                
        self.bconv2    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu2    = nn.LeakyReLU(0.2)                

        # CLS Dec
        self.c_flatten   = nn.Flatten()
        self.c_fc        = nn.utils.spectral_norm(torch.nn.Linear(512, 512, True))
        self.c_relu      = nn.LeakyReLU(0.2)
        self.c_drop      = nn.Dropout(p=0.3)

        # SEG Dec
        self.s_up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu11   = nn.LeakyReLU(0.2)                
        self.s_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu12   = nn.LeakyReLU(0.2)        

        self.s_up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu21   = nn.LeakyReLU(0.2)                
        self.s_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu22   = nn.LeakyReLU(0.2)        

        self.s_up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu31   = nn.LeakyReLU(0.2)                
        self.s_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu32   = nn.LeakyReLU(0.2)        

        self.s_up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu41   = nn.LeakyReLU(0.2)                
        self.s_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu42   = nn.LeakyReLU(0.2)        

        self.s_up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu51   = nn.LeakyReLU(0.2)                
        self.s_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu52   = nn.LeakyReLU(0.2)    

        self.s_up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu61   = nn.LeakyReLU(0.2)                
        self.s_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu62   = nn.LeakyReLU(0.2)    

        # REC Dec
        self.r_up1       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu11   = nn.LeakyReLU(0.2)                
        self.r_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu12   = nn.LeakyReLU(0.2)        

        self.r_up2       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu21   = nn.LeakyReLU(0.2)                
        self.r_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu22   = nn.LeakyReLU(0.2)        

        self.r_up3       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu31   = nn.LeakyReLU(0.2)                
        self.r_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu32   = nn.LeakyReLU(0.2)        

        self.r_up4       = UpsampleBlock(scale=2, input_channels=out_channels*4, output_channels=out_channels*4)
        self.r_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu41   = nn.LeakyReLU(0.2)                
        self.r_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu42   = nn.LeakyReLU(0.2)        

        self.r_up5       = UpsampleBlock(scale=2, input_channels=out_channels*2, output_channels=out_channels*2)
        self.r_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu51   = nn.LeakyReLU(0.2)                
        self.r_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu52   = nn.LeakyReLU(0.2)    

        self.r_up6       = UpsampleBlock(scale=2, input_channels=out_channels, output_channels=out_channels)
        self.r_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu61   = nn.LeakyReLU(0.2)                
        self.r_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu62   = nn.LeakyReLU(0.2)  

        # Heads
        self.enc_out   = nn.Linear(512, 1)
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        self.rec_out   = nn.Conv2d(in_channels, 1, 1)
        
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)


    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.conv11.parameters(),
            self.conv12.parameters(),
            self.down1.parameters(),
            self.conv21.parameters(),
            self.conv22.parameters(),
            self.down2.parameters(),
            self.conv31.parameters(),
            self.conv32.parameters(),
            self.down3.parameters(),
            self.conv41.parameters(),
            self.conv42.parameters(),
            self.down4.parameters(),
            self.conv51.parameters(),
            self.conv52.parameters(),
            self.down5.parameters(),
            self.conv61.parameters(),
            self.conv62.parameters(),
            self.down6.parameters(),
            self.bconv1.parameters(),
            self.bconv2.parameters(),
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.s_dconv11.parameters(),
            self.s_dconv12.parameters(),
            self.s_dconv21.parameters(),
            self.s_dconv22.parameters(),
            self.s_dconv31.parameters(),
            self.s_dconv32.parameters(),
            self.s_dconv41.parameters(),
            self.s_dconv42.parameters(),
            self.s_dconv51.parameters(),
            self.s_dconv52.parameters(),
            self.s_dconv61.parameters(),
            self.s_dconv62.parameters(),
            self.r_up1.parameters(),
            self.r_dconv11.parameters(),
            self.r_dconv12.parameters(),
            self.r_up2.parameters(),
            self.r_dconv21.parameters(),
            self.r_dconv22.parameters(),
            self.r_up3.parameters(),
            self.r_dconv31.parameters(),
            self.r_dconv32.parameters(),
            self.r_up4.parameters(),
            self.r_dconv41.parameters(),
            self.r_dconv42.parameters(),
            self.r_up5.parameters(),
            self.r_dconv51.parameters(),
            self.r_dconv52.parameters(),
            self.r_up6.parameters(),
            self.r_dconv61.parameters(),
            self.r_dconv62.parameters(),
            self.enc_out.parameters(),
            self.dec_out.parameters(),
            self.rec_out.parameters(),
        )

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.bconv2.parameters()


    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv61(x))
        x6 = self.relu62(self.conv62(x))
        x = self.down6(x6)

        # Bottleneck
        x = self.brelu1(self.bconv1(x))
        x_bot = self.brelu2(self.bconv2(x))                
        
        # CLS Decoder
        x = self.c_flatten(x_bot)
        x = self.c_fc(x)
        x = self.c_relu(x)
        cls_out = self.c_drop(x)
        
        # SEG Decoder
        x = self.s_up1(x_bot)
        x = self.s_drelu11(self.s_dconv11(torch.cat([x, x6], dim=1)))
        x = self.s_drelu12(self.s_dconv12(x))                

        x = self.s_up2(x)
        x = self.s_drelu21(self.s_dconv21(torch.cat([x, x5], dim=1)))
        x = self.s_drelu22(self.s_dconv22(x))              

        x = self.s_up3(x)
        x = self.s_drelu31(self.s_dconv31(torch.cat([x, x4], dim=1)))
        x = self.s_drelu32(self.s_dconv32(x))              

        x = self.s_up4(x)
        x = self.s_drelu41(self.s_dconv41(torch.cat([x, x3], dim=1)))
        x = self.s_drelu42(self.s_dconv42(x))              

        x = self.s_up5(x)
        x = self.s_drelu51(self.s_dconv51(torch.cat([x, x2], dim=1)))
        x = self.s_drelu52(self.s_dconv52(x))              

        x = self.s_up6(x)
        x = self.s_drelu61(self.s_dconv61(torch.cat([x, x1], dim=1)))
        seg_out = self.s_drelu62(self.s_dconv62(x))              

        # REC Decoder
        x = self.r_up1(x_bot)
        x = self.r_drelu11(self.r_dconv11(torch.cat([x, x6], dim=1)))
        x = self.r_drelu12(self.r_dconv12(x))                

        x = self.r_up2(x)
        x = self.r_drelu21(self.r_dconv21(torch.cat([x, x5], dim=1)))
        x = self.r_drelu22(self.r_dconv22(x))              

        x = self.r_up3(x)
        x = self.r_drelu31(self.r_dconv31(torch.cat([x, x4], dim=1)))
        x = self.r_drelu32(self.r_dconv32(x))              

        x = self.r_up4(x)
        x = self.r_drelu41(self.r_dconv41(torch.cat([x, x3], dim=1)))
        x = self.r_drelu42(self.r_dconv42(x))              

        x = self.r_up5(x)
        x = self.r_drelu51(self.r_dconv51(torch.cat([x, x2], dim=1)))
        x = self.r_drelu52(self.r_dconv52(x))              
        
        x = self.r_up6(x)
        x = self.r_drelu61(self.r_dconv61(torch.cat([x, x1], dim=1)))
        rec_out = self.r_drelu62(self.r_dconv62(x))       

        # Heads
        x_enc = self.enc_out(cls_out)   
        x_dec = self.dec_out(seg_out)
        x_rec = self.rec_out(rec_out)

        return x_enc, x_dec, x_rec


########################### Ablation Study ###########################
class REDCNN_Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=96, num_layers=10, kernel_size=5, padding=0):
        super(REDCNN_Generator, self).__init__()
        encoder = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        decoder = [nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)]
        for _ in range(num_layers):
            encoder.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
            decoder.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding))
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, x: torch.Tensor):
        residuals = []
        for block in self.encoder:
            residuals.append(x)
            x = F.relu(block(x))
        for residual, block in zip(residuals[::-1], self.decoder[::-1]):
            x = F.relu(block(x) + residual)
        return x
    
class CLS_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv1    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu1    = nn.LeakyReLU(0.2)                
        self.bconv2    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu2    = nn.LeakyReLU(0.2)                

        # CLS Dec
        self.c_flatten   = nn.Flatten()
        self.c_fc        = nn.utils.spectral_norm(torch.nn.Linear(512, 512, True))
        self.c_relu      = nn.LeakyReLU(0.2)
        self.c_drop      = nn.Dropout(p=0.3)

        # Heads
        self.enc_out   = nn.Linear(512, 1)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv61(x))
        x6 = self.relu62(self.conv62(x))
        x = self.down6(x6)
        
        # Bottleneck
        x = self.brelu1(self.bconv1(x))
        x_bot = self.brelu2(self.bconv2(x))                          
        
        # CLS Decoder
        x = self.c_flatten(x_bot)
        x = self.c_fc(x)
        x = self.c_relu(x)
        cls_out = self.c_drop(x)

        # Heads
        x_enc = self.enc_out(cls_out)   

        return x_enc

class SEG_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv1    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu1    = nn.LeakyReLU(0.2)                
        self.bconv2    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu2    = nn.LeakyReLU(0.2)                            

        # SEG Dec
        self.up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu11   = nn.LeakyReLU(0.2)                
        self.dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu12   = nn.LeakyReLU(0.2)        

        self.up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu21   = nn.LeakyReLU(0.2)                
        self.dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.drelu22   = nn.LeakyReLU(0.2)        

        self.up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.drelu31   = nn.LeakyReLU(0.2)                
        self.dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.drelu32   = nn.LeakyReLU(0.2)        

        self.up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.drelu41   = nn.LeakyReLU(0.2)                
        self.dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.drelu42   = nn.LeakyReLU(0.2)        

        self.up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.drelu51   = nn.LeakyReLU(0.2)                
        self.dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.drelu52   = nn.LeakyReLU(0.2)    

        self.up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.drelu61   = nn.LeakyReLU(0.2)                
        self.dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.drelu62   = nn.LeakyReLU(0.2)    

        # Heads
        self.enc_out   = nn.Linear(512, 1)
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv61(x))
        x6 = self.relu62(self.conv62(x))
        x = self.down6(x6)
        
        # Bottleneck
        x = self.brelu1(self.bconv1(x))
        x_bot = self.brelu2(self.bconv2(x))                                

        # SEG Decoder
        x = self.up1(x_bot)
        x = self.drelu11(self.dconv11(torch.cat([x, x6], dim=1)))
        x = self.drelu12(self.dconv12(x))                
        
        x = self.up2(x)
        x = self.drelu21(self.dconv21(torch.cat([x, x5], dim=1)))
        x = self.drelu22(self.dconv22(x))              
        
        x = self.up3(x)
        x = self.drelu31(self.dconv31(torch.cat([x, x4], dim=1)))
        x = self.drelu32(self.dconv32(x))              

        x = self.up4(x)
        x = self.drelu41(self.dconv41(torch.cat([x, x3], dim=1)))
        x = self.drelu42(self.dconv42(x))              

        x = self.up5(x)
        x = self.drelu51(self.dconv51(torch.cat([x, x2], dim=1)))
        x = self.drelu52(self.dconv52(x))              
        
        x = self.up6(x)
        x = self.drelu61(self.dconv61(torch.cat([x, x1], dim=1)))
        seg_out = self.drelu62(self.dconv62(x))              

        # Heads
        x_dec = self.dec_out(seg_out)

        return x_dec

class CLS_SEG_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv1    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu1    = nn.LeakyReLU(0.2)                
        self.bconv2    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu2    = nn.LeakyReLU(0.2)                            

        # CLS Dec
        self.c_flatten   = nn.Flatten()
        self.c_fc        = nn.utils.spectral_norm(torch.nn.Linear(512, 512, True))
        self.c_relu      = nn.LeakyReLU(0.2)
        self.c_drop      = nn.Dropout(p=0.3)

        # SEG Dec
        self.s_up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu11   = nn.LeakyReLU(0.2)                
        self.s_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu12   = nn.LeakyReLU(0.2)        

        self.s_up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu21   = nn.LeakyReLU(0.2)                
        self.s_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu22   = nn.LeakyReLU(0.2)        

        self.s_up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu31   = nn.LeakyReLU(0.2)                
        self.s_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu32   = nn.LeakyReLU(0.2)        

        self.s_up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu41   = nn.LeakyReLU(0.2)                
        self.s_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu42   = nn.LeakyReLU(0.2)        

        self.s_up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu51   = nn.LeakyReLU(0.2)                
        self.s_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu52   = nn.LeakyReLU(0.2)    

        self.s_up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu61   = nn.LeakyReLU(0.2)                
        self.s_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu62   = nn.LeakyReLU(0.2)  

        # Heads
        self.enc_out   = nn.Linear(512, 1)
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv61(x))
        x6 = self.relu62(self.conv62(x))
        x = self.down6(x6)
        
        # Bottleneck
        x = self.brelu1(self.bconv1(x))
        x_bot = self.brelu2(self.bconv2(x))                                

        # CLS Decoder
        x = self.c_flatten(x_bot)
        x = self.c_fc(x)
        x = self.c_relu(x)
        cls_out = self.c_drop(x)

        # SEG Decoder
        x = self.s_up1(x_bot)
        x = self.s_drelu11(self.s_dconv11(torch.cat([x, x6], dim=1)))
        x = self.s_drelu12(self.s_dconv12(x))                

        x = self.s_up2(x)
        x = self.s_drelu21(self.s_dconv21(torch.cat([x, x5], dim=1)))
        x = self.s_drelu22(self.s_dconv22(x))              

        x = self.s_up3(x)
        x = self.s_drelu31(self.s_dconv31(torch.cat([x, x4], dim=1)))
        x = self.s_drelu32(self.s_dconv32(x))              

        x = self.s_up4(x)
        x = self.s_drelu41(self.s_dconv41(torch.cat([x, x3], dim=1)))
        x = self.s_drelu42(self.s_dconv42(x))              

        x = self.s_up5(x)
        x = self.s_drelu51(self.s_dconv51(torch.cat([x, x2], dim=1)))
        x = self.s_drelu52(self.s_dconv52(x))              

        x = self.s_up6(x)
        x = self.s_drelu61(self.s_dconv61(torch.cat([x, x1], dim=1)))
        seg_out = self.s_drelu62(self.s_dconv62(x))                     

        # Heads
        x_enc = self.enc_out(cls_out)   
        x_dec = self.dec_out(seg_out)

        return x_enc, x_dec

class CLS_REC_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv1    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu1    = nn.LeakyReLU(0.2)                
        self.bconv2    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu2    = nn.LeakyReLU(0.2)                

        # CLS Dec
        self.c_flatten   = nn.Flatten()
        self.c_fc        = nn.utils.spectral_norm(torch.nn.Linear(512, 512, True))
        self.c_relu      = nn.LeakyReLU(0.2)
        self.c_drop      = nn.Dropout(p=0.3)

        # REC Dec
        self.r_up1       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu11   = nn.LeakyReLU(0.2)                
        self.r_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu12   = nn.LeakyReLU(0.2)        

        self.r_up2       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu21   = nn.LeakyReLU(0.2)                
        self.r_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu22   = nn.LeakyReLU(0.2)        

        self.r_up3       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu31   = nn.LeakyReLU(0.2)                
        self.r_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu32   = nn.LeakyReLU(0.2)        

        self.r_up4       = UpsampleBlock(scale=2, input_channels=out_channels*4, output_channels=out_channels*4)
        self.r_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu41   = nn.LeakyReLU(0.2)                
        self.r_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu42   = nn.LeakyReLU(0.2)        

        self.r_up5       = UpsampleBlock(scale=2, input_channels=out_channels*2, output_channels=out_channels*2)
        self.r_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu51   = nn.LeakyReLU(0.2)                
        self.r_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu52   = nn.LeakyReLU(0.2)    

        self.r_up6       = UpsampleBlock(scale=2, input_channels=out_channels, output_channels=out_channels)
        self.r_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu61   = nn.LeakyReLU(0.2)                
        self.r_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu62   = nn.LeakyReLU(0.2)  

        # Heads
        self.enc_out   = nn.Linear(512, 1)
        self.rec_out   = nn.Conv2d(in_channels, 1, 1)
        
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv61(x))
        x6 = self.relu62(self.conv62(x))
        x = self.down6(x6)
        
        # Bottleneck
        x = self.brelu1(self.bconv1(x))
        x_bot = self.brelu2(self.bconv2(x))                
        
        # CLS Decoder
        x = self.c_flatten(x_bot)
        x = self.c_fc(x)
        x = self.c_relu(x)
        cls_out = self.c_drop(x)        

        # REC Decoder
        x = self.r_up1(x_bot)
        x = self.r_drelu11(self.r_dconv11(torch.cat([x, x6], dim=1)))
        x = self.r_drelu12(self.r_dconv12(x))                

        x = self.r_up2(x)
        x = self.r_drelu21(self.r_dconv21(torch.cat([x, x5], dim=1)))
        x = self.r_drelu22(self.r_dconv22(x))              

        x = self.r_up3(x)
        x = self.r_drelu31(self.r_dconv31(torch.cat([x, x4], dim=1)))
        x = self.r_drelu32(self.r_dconv32(x))              

        x = self.r_up4(x)
        x = self.r_drelu41(self.r_dconv41(torch.cat([x, x3], dim=1)))
        x = self.r_drelu42(self.r_dconv42(x))              

        x = self.r_up5(x)
        x = self.r_drelu51(self.r_dconv51(torch.cat([x, x2], dim=1)))
        x = self.r_drelu52(self.r_dconv52(x))              
        
        x = self.r_up6(x)
        x = self.r_drelu61(self.r_dconv61(torch.cat([x, x1], dim=1)))
        rec_out = self.r_drelu62(self.r_dconv62(x))       

        # Heads
        x_enc = self.enc_out(cls_out)   
        x_rec = self.rec_out(rec_out)

        return x_enc, x_rec

class SEG_REC_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Enc
        self.conv11    = nn.utils.spectral_norm(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu11    = nn.LeakyReLU(0.2)
        self.conv12    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.relu12    = nn.LeakyReLU(0.2)        
        self.down1     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self.conv21    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu21    = nn.LeakyReLU(0.2)
        self.conv22    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.relu22    = nn.LeakyReLU(0.2)
        self.down2     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=4, stride=2, padding=1))

        self.conv31    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu31    = nn.LeakyReLU(0.2)
        self.conv32    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.relu32    = nn.LeakyReLU(0.2)
        self.down3     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=4, stride=2, padding=1))

        self.conv41    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu41    = nn.LeakyReLU(0.2)
        self.conv42    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu42    = nn.LeakyReLU(0.2)
        self.down4     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        self.conv51    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu51    = nn.LeakyReLU(0.2)
        self.conv52    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu52    = nn.LeakyReLU(0.2)
        self.down5     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))
        
        self.conv61    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu61    = nn.LeakyReLU(0.2)
        self.conv62    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.relu62    = nn.LeakyReLU(0.2)
        self.down6     = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=4, stride=2, padding=1))

        # Bot
        self.bconv1    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu1    = nn.LeakyReLU(0.2)                
        self.bconv2    = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=1, stride=1, padding=0))
        self.brelu2    = nn.LeakyReLU(0.2)                

        # SEG Dec
        self.s_up1       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu11   = nn.LeakyReLU(0.2)                
        self.s_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu12   = nn.LeakyReLU(0.2)        

        self.s_up2       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu21   = nn.LeakyReLU(0.2)                
        self.s_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.s_drelu22   = nn.LeakyReLU(0.2)        

        self.s_up3       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu31   = nn.LeakyReLU(0.2)                
        self.s_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.s_drelu32   = nn.LeakyReLU(0.2)        

        self.s_up4       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu41   = nn.LeakyReLU(0.2)                
        self.s_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.s_drelu42   = nn.LeakyReLU(0.2)        

        self.s_up5       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu51   = nn.LeakyReLU(0.2)                
        self.s_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.s_drelu52   = nn.LeakyReLU(0.2)    

        self.s_up6       = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.s_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu61   = nn.LeakyReLU(0.2)                
        self.s_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.s_drelu62   = nn.LeakyReLU(0.2)    

        # REC Dec
        self.r_up1       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv11   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu11   = nn.LeakyReLU(0.2)                
        self.r_dconv12   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu12   = nn.LeakyReLU(0.2)        

        self.r_up2       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv21   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu21   = nn.LeakyReLU(0.2)                
        self.r_dconv22   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8, out_channels*8, kernel_size=3, stride=1, padding=1))
        self.r_drelu22   = nn.LeakyReLU(0.2)        

        self.r_up3       = UpsampleBlock(scale=2, input_channels=out_channels*8, output_channels=out_channels*8)
        self.r_dconv31   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*8*2, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu31   = nn.LeakyReLU(0.2)                
        self.r_dconv32   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4, out_channels*4, kernel_size=3, stride=1, padding=1))
        self.r_drelu32   = nn.LeakyReLU(0.2)        

        self.r_up4       = UpsampleBlock(scale=2, input_channels=out_channels*4, output_channels=out_channels*4)
        self.r_dconv41   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*4*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu41   = nn.LeakyReLU(0.2)                
        self.r_dconv42   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1))
        self.r_drelu42   = nn.LeakyReLU(0.2)        

        self.r_up5       = UpsampleBlock(scale=2, input_channels=out_channels*2, output_channels=out_channels*2)
        self.r_dconv51   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2*2, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu51   = nn.LeakyReLU(0.2)                
        self.r_dconv52   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.r_drelu52   = nn.LeakyReLU(0.2)    

        self.r_up6       = UpsampleBlock(scale=2, input_channels=out_channels, output_channels=out_channels)
        self.r_dconv61   = nn.utils.spectral_norm(torch.nn.Conv2d(out_channels*2, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu61   = nn.LeakyReLU(0.2)                
        self.r_dconv62   = nn.utils.spectral_norm(torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1))
        self.r_drelu62   = nn.LeakyReLU(0.2)  

        # Heads
        self.dec_out   = nn.Conv2d(in_channels, 1, 1)
        self.rec_out   = nn.Conv2d(in_channels, 1, 1)
        
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:                
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, input):
        # Encoder
        x = self.relu11(self.conv11(input))
        x1 = self.relu12(self.conv12(x))
        x = self.down1(x1)
        
        x = self.relu21(self.conv21(x))
        x2 = self.relu22(self.conv22(x))
        x = self.down2(x2)
        
        x = self.relu31(self.conv31(x))
        x3 = self.relu32(self.conv32(x))
        x = self.down3(x3)
        
        x = self.relu41(self.conv41(x))
        x4 = self.relu42(self.conv42(x))
        x = self.down4(x4)
        
        x = self.relu51(self.conv51(x))
        x5 = self.relu52(self.conv52(x))
        x = self.down5(x5)
        
        x = self.relu61(self.conv61(x))
        x6 = self.relu62(self.conv62(x))
        x = self.down6(x6)
        
        # Bottleneck
        x = self.brelu1(self.bconv1(x))
        x_bot = self.brelu2(self.bconv2(x))                
        
        # SEG Decoder
        x = self.s_up1(x_bot)
        x = self.s_drelu11(self.s_dconv11(torch.cat([x, x6], dim=1)))
        x = self.s_drelu12(self.s_dconv12(x))                

        x = self.s_up2(x)
        x = self.s_drelu21(self.s_dconv21(torch.cat([x, x5], dim=1)))
        x = self.s_drelu22(self.s_dconv22(x))              

        x = self.s_up3(x)
        x = self.s_drelu31(self.s_dconv31(torch.cat([x, x4], dim=1)))
        x = self.s_drelu32(self.s_dconv32(x))              

        x = self.s_up4(x)
        x = self.s_drelu41(self.s_dconv41(torch.cat([x, x3], dim=1)))
        x = self.s_drelu42(self.s_dconv42(x))              

        x = self.s_up5(x)
        x = self.s_drelu51(self.s_dconv51(torch.cat([x, x2], dim=1)))
        x = self.s_drelu52(self.s_dconv52(x))              

        x = self.s_up6(x)
        x = self.s_drelu61(self.s_dconv61(torch.cat([x, x1], dim=1)))
        seg_out = self.s_drelu62(self.s_dconv62(x))              

        # REC Decoder
        x = self.r_up1(x_bot)
        x = self.r_drelu11(self.r_dconv11(torch.cat([x, x6], dim=1)))
        x = self.r_drelu12(self.r_dconv12(x))                

        x = self.r_up2(x)
        x = self.r_drelu21(self.r_dconv21(torch.cat([x, x5], dim=1)))
        x = self.r_drelu22(self.r_dconv22(x))              

        x = self.r_up3(x)
        x = self.r_drelu31(self.r_dconv31(torch.cat([x, x4], dim=1)))
        x = self.r_drelu32(self.r_dconv32(x))              

        x = self.r_up4(x)
        x = self.r_drelu41(self.r_dconv41(torch.cat([x, x3], dim=1)))
        x = self.r_drelu42(self.r_dconv42(x))              

        x = self.r_up5(x)
        x = self.r_drelu51(self.r_dconv51(torch.cat([x, x2], dim=1)))
        x = self.r_drelu52(self.r_dconv52(x))              
        
        x = self.r_up6(x)
        x = self.r_drelu61(self.r_dconv61(torch.cat([x, x1], dim=1)))
        rec_out = self.r_drelu62(self.r_dconv62(x))       

        # Heads
        x_dec = self.dec_out(seg_out)
        x_rec = self.rec_out(rec_out)

        return x_dec, x_rec


# Ablation Single
class Ablation_CLS(nn.Module):
    def __init__(self):
        super(Ablation_CLS, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = CLS_Discriminator(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()   
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake        = self.Generator(x).detach()   
        real_enc    = self.Discriminator(y)
        fake_enc    = self.Discriminator(fake)
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(fake_enc, 0.)

        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss
        loss_details = {'D/real_enc': self.gan_metric(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric(fake_enc, 0.)}

        return total_loss, loss_details

    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc                 = self.Discriminator(fake)

        gen_loss     = self.gan_metric(gen_enc, 1.)

        adv_loss     = gen_loss
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric(gen_enc, 1.),  
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}

        return total_loss, loss_details

class Ablation_SEG(nn.Module):
    def __init__(self):
        super(Ablation_SEG, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = SEG_Discriminator(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()   
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake        = self.Generator(x).detach()   
        real_enc    = self.Discriminator(y)
        fake_enc    = self.Discriminator(fake)
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(fake_enc, 0.)

        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss
        loss_details = {'D/real_enc': self.gan_metric(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric(fake_enc, 0.)}


        return total_loss, loss_details

    def g_loss(self, x, y):
        fake         = self.Generator(x)
        gen_enc      = self.Discriminator(fake)

        gen_loss     = self.gan_metric(gen_enc, 1.)

        adv_loss     = gen_loss
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric(gen_enc, 1.),  
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}

        return total_loss, loss_details


# Ablation Dual
class Ablation_CLS_SEG(nn.Module):
    def __init__(self):
        super(Ablation_CLS_SEG, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = CLS_SEG_Discriminator(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()   
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                  = self.Generator(x).detach()   
        real_enc,  real_dec   = self.Discriminator(y)
        fake_enc,  fake_dec   = self.Discriminator(fake)
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.)

        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss
        loss_details = {'D/real_enc': self.gan_metric(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric(real_dec, 1.),
                        'D/fake_dec': self.gan_metric(fake_dec, 0.)}
        
        return total_loss, loss_details


    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec        = self.Discriminator(fake)

        gen_loss     = self.gan_metric(gen_enc, 1.) + self.gan_metric(gen_dec, 1.)

        adv_loss     = gen_loss
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric(gen_dec, 1.), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        

        return total_loss, loss_details

class Ablation_CLS_REC(nn.Module):
    def __init__(self):
        super(Ablation_CLS_REC, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = CLS_REC_Discriminator(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()   
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                  = self.Generator(x).detach()   
        real_enc,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_rec   = self.Discriminator(fake)
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(fake_enc, 0.)

        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake

        total_loss   = disc_loss + rec_loss
        loss_details = {'D/real_enc': self.gan_metric(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric(fake_enc, 0.), 
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake}
        
        return total_loss, loss_details


    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec        = self.Discriminator(fake)

        gen_loss     = self.gan_metric(gen_enc, 1.) + self.gan_metric(gen_dec, 1.)

        adv_loss     = gen_loss
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric(gen_dec, 1.), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        

        return total_loss, loss_details

class Ablation_SEG_REC(nn.Module):
    def __init__(self):
        super(Ablation_SEG_REC, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = SEG_REC_Discriminator(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()   
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                  = self.Generator(x).detach()   
        real_dec,  real_rec   = self.Discriminator(y)
        fake_dec,  fake_rec   = self.Discriminator(fake)
        
        disc_loss = self.gan_metric(real_dec, 1.) + self.gan_metric(fake_dec, 0.)

        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake

        total_loss   = disc_loss + rec_loss
        loss_details = {'D/real_dec': self.gan_metric(real_dec, 1.),
                        'D/fake_dec': self.gan_metric(fake_dec, 0.),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake}
        
        return total_loss, loss_details


    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec        = self.Discriminator(fake)

        gen_loss     = self.gan_metric(gen_enc, 1.) + self.gan_metric(gen_dec, 1.)

        adv_loss     = gen_loss
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric(gen_dec, 1.), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        

        return total_loss, loss_details


# Ablation Triple
class Ablation_CLS_SEG_REC(nn.Module):
    def __init__(self):
        super(Ablation_CLS_SEG_REC, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = Multi_Task_Discriminator_Skip(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                             = self.Generator(x).detach()   
        real_enc,  real_dec,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_dec,  fake_rec   = self.Discriminator(fake)
        
        disc_loss = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.)

        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake

        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss + rec_loss
        loss_details = {'D/real_enc': self.gan_metric(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric(real_dec, 1.),
                        'D/fake_dec': self.gan_metric(fake_dec, 0.),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake}
        
        return total_loss, loss_details


    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec, _     = self.Discriminator(fake)

        adv_loss     = self.gan_metric(gen_enc, 1.) + self.gan_metric(gen_dec, 1.)
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric(gen_dec, 1.), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        

        return total_loss, loss_details

# + NDS regulation
class Ablation_CLS_SEG_REC_NDS(nn.Module):
    def __init__(self):
        super(Ablation_CLS_SEG_REC_NDS, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = Multi_Task_Discriminator_Skip(in_channels=1, out_channels=64)
        

        # LOSS
        self.gan_metric_cls  = ls_gan
        self.gan_metric_seg  = NDS_Loss
        
        self.pixel_loss     = CharbonnierLoss()
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                             = self.Generator(x).detach()   
        real_enc,  real_dec,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_dec,  fake_rec   = self.Discriminator(fake)

        disc_loss    = self.gan_metric_cls(real_enc, 1.) + self.gan_metric_cls(fake_enc, 0.) + self.gan_metric_seg(real_dec, 1., x-y) + self.gan_metric_seg(fake_dec, 0., x-y)

        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake
        
        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss + rec_loss
        loss_details = {'D/real_enc': self.gan_metric_cls(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric_cls(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric_seg(real_dec, 1., x-y),
                        'D/fake_dec': self.gan_metric_seg(fake_dec, 0., x-y),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake}
        
        return total_loss, loss_details


    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec, _     = self.Discriminator(fake)

        adv_loss     = self.gan_metric_cls(gen_enc, 1.) + self.gan_metric_seg(gen_dec, 1., x-y)
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric_cls(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric_seg(gen_dec, 1., x-y), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        
        return total_loss, loss_details

# + RC regulation
class Ablation_CLS_SEG_REC_RC(nn.Module):
    def __init__(self):
        super(Ablation_CLS_SEG_REC_RC, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = Multi_Task_Discriminator_Skip(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric      = ls_gan
        
        self.pixel_loss     = CharbonnierLoss()
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                             = self.Generator(x).detach()   
        real_enc,  real_dec,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_dec,  fake_rec   = self.Discriminator(fake)
    
        disc_loss  = self.gan_metric(real_enc, 1.) + self.gan_metric(real_dec, 1.) + self.gan_metric(fake_enc, 0.) + self.gan_metric(fake_dec, 0.)

        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake
        
        # Consistency
        rec_real_enc,  rec_real_dec,  _   = self.Discriminator(real_rec.clip(0, 1))
        rec_fake_enc,  rec_fake_dec,  _   = self.Discriminator(fake_rec.clip(0, 1))

        consist_loss_real_enc = F.mse_loss(real_enc, rec_real_enc) 
        consist_loss_real_dec = F.mse_loss(real_dec, rec_real_dec)
        consist_loss_fake_enc = F.mse_loss(fake_enc, rec_fake_enc) 
        consist_loss_fake_dec = F.mse_loss(fake_dec, rec_fake_dec)

        consist_loss = consist_loss_real_enc + consist_loss_real_dec + consist_loss_fake_enc + consist_loss_fake_dec
        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss + rec_loss + consist_loss
        loss_details = {'D/real_enc': self.gan_metric(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric(real_dec, 1.),
                        'D/fake_dec': self.gan_metric(fake_dec, 0.),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake,
                        'D/consist_loss_real_enc': consist_loss_real_enc,
                        'D/consist_loss_real_dec': consist_loss_real_dec,
                        'D/consist_loss_fake_enc': consist_loss_fake_enc,
                        'D/consist_loss_fake_dec': consist_loss_fake_dec}     

        return total_loss, loss_details
        

    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec, _     = self.Discriminator(fake)
        
        adv_loss     = self.gan_metric(gen_enc, 1.) + self.gan_metric(gen_dec, 1.)
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric(gen_dec, 1.), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        
        return total_loss, loss_details
 
# + NDS + RC regulations
class Ablation_CLS_SEG_REC_NDS_RC(nn.Module):
    def __init__(self):
        super(Ablation_CLS_SEG_REC_NDS_RC, self).__init__()
        # Generator
        self.Generator       = REDCNN_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = Multi_Task_Discriminator_Skip(in_channels=1, out_channels=64)

        # LOSS
        self.gan_metric_cls  = ls_gan
        self.gan_metric_seg  = NDS_Loss
        
        self.pixel_loss     = CharbonnierLoss()
        self.edge_loss      = EdgeLoss()

    def d_loss(self, x, y):
        fake                             = self.Generator(x).detach()   
        real_enc,  real_dec,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_dec,  fake_rec   = self.Discriminator(fake)
    
        disc_loss    = self.gan_metric_cls(real_enc, 1.) + self.gan_metric_cls(fake_enc, 0.) + self.gan_metric_seg(real_dec, 1., x-y) + self.gan_metric_seg(fake_dec, 0., x-y)
        
        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake
        
        # Consistency
        rec_real_enc,  rec_real_dec,  _   = self.Discriminator(real_rec.clip(0, 1))
        rec_fake_enc,  rec_fake_dec,  _   = self.Discriminator(fake_rec.clip(0, 1))

        consist_loss_real_enc = F.mse_loss(real_enc, rec_real_enc) 
        consist_loss_real_dec = F.mse_loss(real_dec, rec_real_dec)
        consist_loss_fake_enc = F.mse_loss(fake_enc, rec_fake_enc) 
        consist_loss_fake_dec = F.mse_loss(fake_dec, rec_fake_dec)

        consist_loss = consist_loss_real_enc + consist_loss_real_dec + consist_loss_fake_enc + consist_loss_fake_dec
        print("D / real_enc == ", real_enc.max())
        print("D / fake_enc == ", fake_enc.max())

        total_loss   = disc_loss + rec_loss + consist_loss
        loss_details = {'D/real_enc': self.gan_metric_cls(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric_cls(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric_seg(real_dec, 1., x-y),
                        'D/fake_dec': self.gan_metric_seg(fake_dec, 0., x-y),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake,
                        'D/consist_loss_real_enc': consist_loss_real_enc,
                        'D/consist_loss_real_dec': consist_loss_real_dec,
                        'D/consist_loss_fake_enc': consist_loss_fake_enc,
                        'D/consist_loss_fake_dec': consist_loss_fake_dec}     

        return total_loss, loss_details
        

    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec, _     = self.Discriminator(fake)
        
        adv_loss     = self.gan_metric_cls(gen_enc, 1.) + self.gan_metric_seg(gen_dec, 1., x-y)
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        print("G / real_enc == ", gen_enc.max())

        total_loss   = adv_loss + pix_loss + edge_loss
        loss_details = {'G/gen_enc': self.gan_metric_cls(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric_seg(gen_dec, 1., x-y), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        
        return total_loss, loss_details
 
# + NDS + RC regulations + Res-FFT-Conv Generator
class Ablation_CLS_SEG_REC_NDS_RC_ResFFT(nn.Module):
    def __init__(self):
        super(Ablation_CLS_SEG_REC_NDS_RC_ResFFT, self).__init__()
        # Generator
        self.Generator       = ResFFT_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = Multi_Task_Discriminator_Skip(in_channels=1, out_channels=64)
        
        # LOSS
        self.gan_metric_cls  = ls_gan
        self.gan_metric_seg  = NDS_Loss
        
        self.pixel_loss     = CharbonnierLoss()
        self.edge_loss      = EdgeLoss()

    # Both REC
    def d_loss(self, x, y):
        fake                             = self.Generator(x).detach()   
        real_enc,  real_dec,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_dec,  fake_rec   = self.Discriminator(fake)
    
        disc_loss    = self.gan_metric_cls(real_enc, 1.) + self.gan_metric_cls(fake_enc, 0.) + self.gan_metric_seg(real_dec, 1., x-y) + self.gan_metric_seg(fake_dec, 0., x-y)
        
        rec_loss_real     = F.l1_loss(real_rec, y) 
        rec_loss_fake     = F.l1_loss(fake_rec, fake) 
        rec_loss          = rec_loss_real + rec_loss_fake

        # Consistency
        rec_real_enc,  rec_real_dec,  _   = self.Discriminator(real_rec.clip(0, 1))
        rec_fake_enc,  rec_fake_dec,  _   = self.Discriminator(fake_rec.clip(0, 1))

        consist_loss_real_enc = F.mse_loss(real_enc, rec_real_enc) 
        consist_loss_real_dec = F.mse_loss(real_dec, rec_real_dec)
        consist_loss_fake_enc = F.mse_loss(fake_enc, rec_fake_enc) 
        consist_loss_fake_dec = F.mse_loss(fake_dec, rec_fake_dec)

        consist_loss = consist_loss_real_enc + consist_loss_real_dec + consist_loss_fake_enc + consist_loss_fake_dec

        total_loss   = disc_loss + rec_loss + consist_loss

        loss_details = {'D/real_enc': self.gan_metric_cls(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric_cls(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric_seg(real_dec, 1., x-y),
                        'D/fake_dec': self.gan_metric_seg(fake_dec, 0., x-y),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake,
                        'D/consist_loss_real_enc': consist_loss_real_enc,
                        'D/consist_loss_real_dec': consist_loss_real_dec,
                        'D/consist_loss_fake_enc': consist_loss_fake_enc,
                        'D/consist_loss_fake_dec': consist_loss_fake_dec}     

        return total_loss, loss_details
    
    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec, _     = self.Discriminator(fake)
        
        adv_loss     = self.gan_metric_cls(gen_enc, 1.) + self.gan_metric_seg(gen_dec, 1., x-y)
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        total_loss   = adv_loss + pix_loss + edge_loss
        
        loss_details = {'G/gen_enc': self.gan_metric_cls(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric_seg(gen_dec, 1., x-y), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}

        return total_loss, loss_details


# MTD-GAN Final
class MTD_GAN_Method(nn.Module):
    def __init__(self):
        super(MTD_GAN_Method, self).__init__()
        # Generator
        self.Generator       = ResFFT_Generator(in_channels=1, out_channels=32, num_layers=10, kernel_size=3, padding=1)

        # Discriminator
        self.Discriminator   = Multi_Task_Discriminator_Skip(in_channels=1, out_channels=64)
        
        # LOSS
        self.gan_metric_cls  = ls_gan
        self.gan_metric_seg  = NDS_Loss
        
        self.pixel_loss     = CharbonnierLoss()
        self.edge_loss      = EdgeLoss()

    # Both REC
    def d_loss(self, x, y):
        fake                             = self.Generator(x).detach()   
        real_enc,  real_dec,  real_rec   = self.Discriminator(y)
        fake_enc,  fake_dec,  fake_rec   = self.Discriminator(fake)
    
        disc_loss    = self.gan_metric_cls(real_enc, 1.) + self.gan_metric_cls(fake_enc, 0.) + self.gan_metric_seg(real_dec, 1., x-y) + self.gan_metric_seg(fake_dec, 0., x-y)
        
        rec_loss_real = F.l1_loss(real_rec, y) 
        rec_loss_fake = F.l1_loss(fake_rec, fake) 
        rec_loss      = rec_loss_real + rec_loss_fake

        # Consistency
        rec_real_enc,  rec_real_dec,  _   = self.Discriminator(real_rec.clip(0, 1))
        rec_fake_enc,  rec_fake_dec,  _   = self.Discriminator(fake_rec.clip(0, 1))

        consist_loss_real_enc = F.mse_loss(real_enc, rec_real_enc) 
        consist_loss_real_dec = F.mse_loss(real_dec, rec_real_dec)
        consist_loss_fake_enc = F.mse_loss(fake_enc, rec_fake_enc) 
        consist_loss_fake_dec = F.mse_loss(fake_dec, rec_fake_dec)

        consist_loss = consist_loss_real_enc + consist_loss_real_dec + consist_loss_fake_enc + consist_loss_fake_dec

        loss_details = {'D/real_enc': self.gan_metric_cls(real_enc, 1.), 
                        'D/fake_enc': self.gan_metric_cls(fake_enc, 0.), 
                        'D/real_dec': self.gan_metric_seg(real_dec, 1., x-y),
                        'D/fake_dec': self.gan_metric_seg(fake_dec, 0., x-y),
                        'D/rec_loss_real': rec_loss_real,
                        'D/rec_loss_fake': rec_loss_fake,
                        'D/consist_loss_real_enc': consist_loss_real_enc,
                        'D/consist_loss_real_dec': consist_loss_real_dec,
                        'D/consist_loss_fake_enc': consist_loss_fake_enc,
                        'D/consist_loss_fake_dec': consist_loss_fake_dec}     

        return torch.stack([disc_loss, rec_loss, consist_loss]), loss_details
    
    def g_loss(self, x, y):
        fake                    = self.Generator(x)
        gen_enc, gen_dec, _     = self.Discriminator(fake)
        
        adv_loss     = self.gan_metric_cls(gen_enc, 1.) + self.gan_metric_seg(gen_dec, 1., x-y)
        pix_loss     = 50.0*self.pixel_loss(fake, y)
        edge_loss    = 50.0*self.edge_loss(fake, y)

        total_loss   = adv_loss + pix_loss + edge_loss

        loss_details = {'G/gen_enc': self.gan_metric_cls(gen_enc, 1.), 
                        'G/gen_dec': self.gan_metric_seg(gen_dec, 1., x-y), 
                        'G/pix_loss': pix_loss,
                        'G/edge_loss': edge_loss}
                        
        return total_loss, loss_details

