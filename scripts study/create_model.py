import torch 
import torch.nn as nn
import torch.nn.functional as F
import sys
import os 

sys.path.append(os.path.abspath('/workspace/sunggu'))
sys.path.append(os.path.abspath('/workspace/sunggu/4.Dose_img2img/utils'))


######################################################################################################################################################################
######################################################                    Create Model                        ########################################################
######################################################################################################################################################################

# Previous Works
from RED_CNN.networks import RED_CNN
from EDCNN.code.edcnn_model import EDCNN
from SACNN.model import SACNN, AutoEncoder_2D
from WGAN_VGG.networks import WGAN_VGG
from DU_GAN.models.DUGAN.networks import DUGAN, FSGAN
from MAP_NN.models import MAP_NN
from TED_net.t2t_shortcuts import TED_Net, TED_Net_Mixed
from Restormer_module.Restormer import Restormer
# Ours Works
from Ours.MLPmixer import MLPMixer, Img2Img_Mixer, HF_ConvMixer
from Ours.models import *
from Unet_sunggu.model import *

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
        ### Base
    elif name == "FSGAN":
        model = FSGAN()     
        ### Ablation Version

                                                           
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model


