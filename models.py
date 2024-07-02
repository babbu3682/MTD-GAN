import torch 
import torch.nn as nn
import torch.nn.functional as F


# Create Model
from arch.RED_CNN.networks import RED_CNN
from arch.EDCNN.networks import EDCNN
from arch.CTformer.networks import CTformer
from arch.Restormer.networks import Restormer
from arch.Diffusion.networks import DDPM, DDIM, PNDM, DPM
from arch.WGAN_VGG.networks import WGAN_VGG
from arch.MAP_NN.networks import MAP_NN
from arch.DUGAN.networks import DUGAN
from arch.Ours.networks import *



def get_model(name):
    
    # CNN-based models
    if name == "RED_CNN":
        model = RED_CNN()
    elif name == "EDCNN":
        model = EDCNN()

    # TR-based models
    elif name == "CTformer":
        model = CTformer(img_size=64, tokens_type='performer', embed_dim=64, depth=1, num_heads=8, kernel=4, stride=4, mlp_ratio=2., token_dim=64)
    elif name == "Restormer":
        model = Restormer(LayerNorm_type='BiasFree')
    
    # GAN-based models
    elif name == "WGAN_VGG":
        model = WGAN_VGG()
    elif name == "MAP_NN" or name == "MAP_NN_brain":
        model = MAP_NN()
    elif name == "DU_GAN" or name == "DU_GAN_brain":
        model = DUGAN()

    # DN-based models
    elif name == "DDPM":
        model = DDPM()
    elif name == "DDIM":
        model = DDIM()
    elif name == "PNDM":
        model = PNDM()        
    elif name == "DPM":
        model = DPM()

    # Ours
    elif name == "MTD_GAN_Method":
        model = MTD_GAN_Method()

    # Ablation studies
    elif name == "Ablation_CLS":
        model = Ablation_CLS()
    elif name == "Ablation_SEG":
        model = Ablation_SEG()
    elif name == "Ablation_CLS_SEG":
        model = Ablation_CLS_SEG()
    elif name == "Ablation_CLS_REC":
        model = Ablation_CLS_REC()
    elif name == "Ablation_SEG_REC":
        model = Ablation_SEG_REC()
    elif name == "Ablation_CLS_SEG_REC":
        model = Ablation_CLS_SEG_REC()
    elif name == "Ablation_CLS_SEG_REC_NDS":
        model = Ablation_CLS_SEG_REC_NDS()
    elif name == "Ablation_CLS_SEG_REC_RC":
        model = Ablation_CLS_SEG_REC_RC()
    elif name == "Ablation_CLS_SEG_REC_NDS_RC":
        model = Ablation_CLS_SEG_REC_NDS_RC()
    elif name == "Ablation_CLS_SEG_REC_NDS_RC_ResFFT":
        model = Ablation_CLS_SEG_REC_NDS_RC_ResFFT()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model


