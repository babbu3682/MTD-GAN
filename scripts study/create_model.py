import torch 
import torch.nn as nn
import torch.nn.functional as F
import sys
import os 

sys.path.append(os.path.abspath('/workspace/sunggu'))
sys.path.append(os.path.abspath('/workspace/sunggu/4.Dose_img2img/LowDose_HighDose_Code_Factory/utils'))


######################################################################################################################################################################
######################################################                    Create Model                        ########################################################
######################################################################################################################################################################

# Previous Works
from RED_CNN.networks import RED_CNN
from EDCNN.code.edcnn_model import EDCNN
from SACNN.model import SACNN, AutoEncoder_2D
from WGAN_VGG.networks import WGAN_VGG
from DUGAN.networks import DUGAN
from MAP_NN.models import MAP_NN
from TED_net.t2t_shortcuts import TED_Net, TED_Net_Mixed
from Restormer_module.Restormer import Restormer
# Ours Works
from Ours.MLPmixer import MLPMixer, Img2Img_Mixer, HF_ConvMixer
from Ours.network import *
from Ours.Unet_Factory import *

def create_model(name):
    if name == "SACNN_AutoEncoder":
        model = AutoEncoder_2D()    
    
    # CNN based
        ### Previous
    elif name == "RED_CNN":
        model = RED_CNN()
    elif name == "ED_CNN":
        model = EDCNN()
    elif name == "TED_Net":
        model = TED_Net(img_size=64, tokens_type='performer', embed_dim=512, depth=1, num_heads=8, kernel=8, stride=4, mlp_ratio=2., token_dim=64)
    elif name == "Restormer":
        model = Restormer(LayerNorm_type='BiasFree')    
    
    # GAN based
        ### Previous
    elif name == "WGAN_VGG":
        model = WGAN_VGG()
    elif name == "MAP_NN":
        model = MAP_NN()
    elif name == "SACNN":
        model = SACNN()
    elif name == "DU_GAN":
        model = DUGAN()

    # Ours
        ### CNN Base
    elif name == "SPADE_UNet":
        model = SPADE_UNet()

    elif name == "SPADE_UNet_Upgrade":
        model = SPADE_UNet_Upgrade()

        ### GAN Base
    # elif name == "FSGAN":
    #     model = FSGAN(generator_type="ConvMixer")
        
        ## Ablation Version
    # elif name == "FSGAN":
    #     model = FSGAN(generator_type="Restormer")

    elif name == "FSGAN":
        model = FSGAN(generator_type="Restormer_Decoder")    

    # elif name == "FSGAN":
    #     model = FSGAN(generator_type="Uformer_Decoder")            

    # 1. TEST Unet vs Unet GAN
    # 2. TEST Restomer vs Unet 
    # 3. TEST 해상도 유지 SPADE vs 업샘플 (Transformer_Generator vs Restormer_Decoder/Uformer_Decoder)        
    elif name == "Revised_UNet":
        model = Revised_UNet()

    elif name == "Unet_GAN":
        model = Unet_GAN()        

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model


