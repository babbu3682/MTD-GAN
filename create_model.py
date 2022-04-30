import torch 
import torch.nn as nn
import torch.nn.functional as F


# Create Model
    # Previous Works
from arch.RED_CNN.networks import RED_CNN
from arch.EDCNN.code.edcnn_model import EDCNN
from arch.TED_net.t2t_shortcuts import TED_Net, TED_Net_Mixed
from arch.Restormer_module.Restormer import Restormer
from arch.WGAN_VGG.networks import WGAN_VGG
from arch.MAP_NN.models import MAP_NN
from arch.Markovian_Patch_GAN.networks import Markovian_Patch_GAN
from arch.DUGAN.networks import DUGAN
# from arch.SACNN.model import SACNN, AutoEncoder_2D
    # Ours Works
from arch.Ours.network import ResFFT_LFSPADE, ResFFT_Freq_SPADE_Att, FSGAN, FDGAN_PatchGAN



def create_model(name):    
    # Previous
        ### CNN based
    if name == "RED_CNN":
        model = RED_CNN()
    elif name == "ED_CNN":
        model = EDCNN()
    elif name == "TED_Net":
        model = TED_Net(img_size=64, tokens_type='performer', embed_dim=512, depth=1, num_heads=8, kernel=8, stride=4, mlp_ratio=2., token_dim=64)
    elif name == "Restormer":
        model = Restormer(LayerNorm_type='BiasFree')    
    
        ### GAN based
    elif name == "WGAN_VGG":
        model = WGAN_VGG()
    elif name == "MAP_NN":
        model = MAP_NN()
    elif name == "DU_GAN":
        model = DUGAN()
    elif name == "Markovian_Patch_GAN":
        model = Markovian_Patch_GAN()

    # elif name == "SACNN":
    #     model = SACNN()        


    # Ours
        ### CNN Base
    elif name == "ResFFT_LFSPADE":
        model = ResFFT_LFSPADE()

    elif name == "ResFFT_Freq_SPADE_Att" or name == "ResFFT_Freq_SPADE_Att_window":
        model = ResFFT_Freq_SPADE_Att()

        ### GAN Base
    elif name == "FSGAN":
        model = FSGAN(generator_type="Restormer_Decoder")    

    elif name == "FDGAN_PatchGAN":
        model = FDGAN_PatchGAN()    


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model


