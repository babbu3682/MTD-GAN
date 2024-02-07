import torch
import numpy as np
from monai.transforms import *
from monai.data import Dataset
import re
import glob
import pydicom
import functools

# import warnings
# warnings.filterwarnings(action='ignore') 


def list_sort_nicely(l):   
    def tryint(s):        
        try:            
            return int(s)        
        except:            
            return s
        
    def alphanum_key(s):
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)    
    return l

def get_pixels_hu(path):
    # pydicom version...!
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    # ref: pydicom.pixel_data_handlers.util.apply_modality_lut
    # '''
    # Awesome pydicom lut fuction...!
    # ds  = pydicom.dcmread(fname)
    # arr = ds.pixel_array
    # hu  = apply_modality_lut(arr, ds)
    # '''
    dcm_image = pydicom.read_file(path)
    image = dcm_image.pixel_array
    image = image.astype(np.int16)
    image[image == -2000] = 0

    intercept = dcm_image.RescaleIntercept
    slope     = dcm_image.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    # print(image.shape) # (512, 512)
    return np.array(image, dtype=np.int16)

def dicom_normalize(image, MIN_HU=-1024.0, MAX_HU=3072.0):   # I already check the max value is 3072.0
   image = (image - MIN_HU) / (MAX_HU - MIN_HU)   # Range  0.0 ~ 1.0
   # image = (image - 0.5) / 0.5                  # Range -1.0 ~ 1.0   @ We do not use -1~1 range becuase there is no Tanh act.
   return image

def minmax_normalize(image, option=False):
    if len(np.unique(image)) != 1:  # Sometimes it cause the nan inputs...
        image -= image.min()
        image /= image.max() 

    if option:
        image = (image - 0.5) / 0.5  # Range -1.0 ~ 1.0   @ We do not use -1~1 range becuase there is no Tanh act.

    return image.astype('float32')


# collate_fn
def default_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



# Follow Task
def Follow_Dataset_DCM(mode, patch_training):
    if mode == 'train':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/20/*/*/*.dcm')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/20/*/*/*.dcm'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/X/*/*/*.dcm'))  + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/X/*/*/*.dcm'))

        files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]            
        print("Train [Total]  number = ", len(n_20_imgs))

        if patch_training:
            transforms = Compose(
                [
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20", "n_100"]),    

                    # Crop  
                    RandSpatialCropSamplesd(keys=["n_20", "n_100"], roi_size=(64, 64), num_samples=8, random_center=True, random_size=False, meta_keys=None, allow_missing_keys=False), 
                        # patch training, next(iter(loader)) output : list로 sample 만큼,,, 그 List 안에 (B, C, H, W)

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    # RandZoomd(keys=["n_20", "n_100"], prob=0.1, min_zoom=0.8, max_zoom=1.2, align_corners=None, keep_size=True, allow_missing_keys=False), # whole image 일때는 괜찮지만, Patch를 뜯을때 사용하면, 치명적이다...

                    # Normalize
                    # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),     
                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )  

        else :
            transforms = Compose(
                [
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20", "n_100"]),                 

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    # RandZoomd(keys=["n_20", "n_100"], prob=0.1, min_zoom=0.8, max_zoom=1.2, align_corners=None, keep_size=True, allow_missing_keys=False), # whole image 일때는 괜찮지만, Patch를 뜯을때 사용하면, 치명적이다...
                    
                    # Normalize
                    # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )       


    elif mode == 'valid':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/20/*/*/*.dcm'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/X/*/*/*.dcm'))        

        files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]
        print("Valid [Total]  number = ", len(n_20_imgs))

        # CT에 맞는 Augmentation
        transforms = Compose(
            [
                Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                AddChanneld(keys=["n_20", "n_100"]),     

                # Normalize
                # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                ToTensord(keys=["n_20", "n_100"]),
            ]
        )    


    return Dataset(data=files, transform=transforms), default_collate_fn

def Follow_Dataset_DCM_Windowing(mode, patch_training):
    if mode == 'train':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/20/*/*/*.dcm')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/20/*/*/*.dcm'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/X/*/*/*.dcm'))  + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/X/*/*/*.dcm'))

        files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]            
        print("Train [Total]  number = ", len(n_20_imgs))

        if patch_training:
            transforms = Compose(
                [
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]             
                        # Only for WGAN-VGG 
                    # ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=-40.0, a_max=120.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]             
                    
                    AddChanneld(keys=["n_20", "n_100"]),    
                    
                    # Crop  
                    CropForegroundd(keys=["n_20", "n_100"], source_key="n_100", select_fn=lambda x: x > 0),
                    SpatialPadd(keys=["n_20", "n_100"], spatial_size=(64, 64)),
                    RandSpatialCropSamplesd(keys=["n_20", "n_100"], roi_size=(64, 64), num_samples=8, random_center=True, random_size=False, meta_keys=None, allow_missing_keys=False), 
                        # patch training, next(iter(loader)) output : list로 sample 만큼,,, 그 List 안에 (B, C, H, W)

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    # RandZoomd(keys=["n_20", "n_100"], prob=0.1, min_zoom=0.8, max_zoom=1.2, align_corners=None, keep_size=True, allow_missing_keys=False), # whole image 일때는 괜찮지만, Patch를 뜯을때 사용하면, 치명적이다...

                    # Normalize
                    # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),     
                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )  



        else :
            transforms = Compose(
                [
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]             
                        # Only for WGAN-VGG 
                    # ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=-40.0, a_max=120.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]                                 
                    
                    AddChanneld(keys=["n_20", "n_100"]),                 

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    # RandZoomd(keys=["n_20", "n_100"], prob=0.1, min_zoom=0.8, max_zoom=1.2, align_corners=None, keep_size=True, allow_missing_keys=False), # whole image 일때는 괜찮지만, Patch를 뜯을때 사용하면, 치명적이다...
                    
                    # Normalize
                    # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )              

    elif mode == 'valid':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/20/*/*/*.dcm'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/X/*/*/*.dcm'))        

        files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]
        print("Valid [Total]  number = ", len(n_20_imgs))

        # CT에 맞는 Augmentation
        transforms = Compose(
            [
                Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]             
                    # Only for WGAN-VGG 
                # ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=-40.0, a_max=120.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]                                                 
               
                AddChanneld(keys=["n_20", "n_100"]),     

                # Normalize
                # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                ToTensord(keys=["n_20", "n_100"]),
            ]
        )    


    return Dataset(data=files, transform=transforms), default_collate_fn




# TEST   Follow Task
def TEST_Follow_Dataset_DCM():
    low_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/Brain_Dose_Follow_dataset/dcm_original_dataset/*/*CT, Low Dose Neck With Enhance/*.dcm'))
    high_imgs = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/Brain_Dose_Follow_dataset/dcm_original_dataset/*/*CT, Neck Other With Enhance/*.dcm'))

    files = [{"n_20": low_name, "n_100": high_name, "path_n_20":low_path, "path_n_100":high_path} for low_name, high_name, low_path, high_path in zip(low_imgs, high_imgs, low_imgs, high_imgs)]
    print("TEST [Total]  number = ", len(low_imgs))

    transforms = Compose(
        [
            Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
            Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
            AddChanneld(keys=["n_20", "n_100"]),         

            # Normalize
            # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                         
            ToTensord(keys=["n_20", "n_100"]),
        ]
    )        

    return Dataset(data=files, transform=transforms), default_collate_fn

def TEST_Follow_Dataset_DCM_Windowing():
    low_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/Brain_Dose_Follow_dataset/dcm_original_dataset/*/*CT, Low Dose Neck With Enhance/*.dcm'))
    high_imgs = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/Brain_Dose_Follow_dataset/dcm_original_dataset/*/*CT, Neck Other With Enhance/*.dcm'))

    files = [{"n_20": low_name, "n_100": high_name, "path_n_20":low_path, "path_n_100":high_path} for low_name, high_name, low_path, high_path in zip(low_imgs, high_imgs, low_imgs, high_imgs)]
    print("TEST [Total]  number = ", len(low_imgs))

    transforms = Compose(
        [
            Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
            ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),          # No Margin
            # ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=-40.0, a_max=120.0, b_min=0.0, b_max=1.0, clip=True),     # Yes Margin
            AddChanneld(keys=["n_20", "n_100"]),         

            # Normalize
            # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                         
            ToTensord(keys=["n_20", "n_100"]),
        ]
    )        

    return Dataset(data=files, transform=transforms), default_collate_fn
