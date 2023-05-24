import torch 
import torch.nn as nn
import torch.nn.functional as F


# Create Model
    # Previous Works
from arch.RED_CNN.networks import RED_CNN
from arch.EDCNN.code.edcnn_model import EDCNN
from arch.CTformer.CTformer import CTformer
from arch.Restormer_module.Restormer import Restormer
from arch.WGAN_VGG.networks import WGAN_VGG
from arch.MAP_NN.models import MAP_NN
from arch.Markovian_Patch_GAN.networks import Markovian_Patch_GAN
from arch.DUGAN.networks import DUGAN

    # Ours Works
from arch.Ours.network import FDGAN, FDGAN_domain, MTD_GAN, Ablation_A, Ablation_B, Ablation_C, Ablation_D, Ablation_D2, Ablation_E



def create_model(name):
    # Previous
        ### CNN based
    if name == "RED_CNN":
        model = RED_CNN()
    elif name == "ED_CNN":
        model = EDCNN()
    elif name == "CTformer":
        model = CTformer(img_size=64, tokens_type='performer', embed_dim=64, depth=1, num_heads=8, kernel=4, stride=4, mlp_ratio=2., token_dim=64)
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

    # Ours
        ### CNN Base
    elif name == "ResFFT_LFSPADE":
        model = ResFFT_LFSPADE()

    elif name == "ResFFT_Freq_SPADE_Att" or name == "ResFFT_Freq_SPADE_Att_window":
        model = ResFFT_Freq_SPADE_Att()

        ### GAN Base
    elif name == "FDGAN":
        model = FDGAN()

    elif name == "FDGAN_domain":
        model = FDGAN_domain()

    elif name == "MTD_GAN":
        model = MTD_GAN()  

    elif name == "Ablation_A":
        model = Ablation_A()        

    elif name == "Ablation_B":
        model = Ablation_B()       

    elif name == "Ablation_C":
        model = Ablation_C()       

    elif name == "Ablation_D":
        model = Ablation_D()       

    elif name == "Ablation_D2":
        model = Ablation_D2()       

    elif name == "Ablation_E":
        model = Ablation_E()       
             


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model


