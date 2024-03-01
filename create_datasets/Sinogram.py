import torch
import numpy as np
from monai.transforms import *
from monai.data import Dataset, list_data_collate
import re
import glob
import pydicom
import functools

# import warnings
# warnings.filterwarnings(action='ignore') 


def list_sort_nicely(l):
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

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

def default_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_transforms(mode="train", type="full_patch"):
    # Training Augmentation
    if mode == "train":
        if type == "full":
            return Compose([
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20", "n_100"]),                 

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    
                    # Normalize
                    # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                    ToTensord(keys=["n_20", "n_100"]),
                ])       

        elif type == "full_patch":
            return Compose([
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20", "n_100"]),    

                    # Crop, patch training, next(iter(loader)) output : list로 sample 만큼,,, 그 List 안에 (B, C, H, W)
                    RandSpatialCropSamplesd(keys=["n_20", "n_100"], roi_size=(64, 64), num_samples=8, random_center=True, random_size=False, allow_missing_keys=False), 

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),

                    # Normalize
                    # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),     
                    ToTensord(keys=["n_20", "n_100"])
                ])

        elif type == "window":
            return Compose([
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]
                    AddChanneld(keys=["n_20", "n_100"]),                 

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    
                    # Normalize
                    # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                    ToTensord(keys=["n_20", "n_100"]),
                ]) 

        elif type == "window_patch":
            return Compose([
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]
                    AddChanneld(keys=["n_20", "n_100"]),    
                    
                    # Crop, patch training, next(iter(loader)) output : list로 sample 만큼,,, 그 List 안에 (B, C, H, W)
                    CropForegroundd(keys=["n_20", "n_100"], source_key="n_100", select_fn=lambda x: x > 0),
                    SpatialPadd(keys=["n_20", "n_100"], spatial_size=(64, 64)),
                    RandSpatialCropSamplesd(keys=["n_20", "n_100"], roi_size=(64, 64), num_samples=8, random_center=True, random_size=False, allow_missing_keys=False), 
                        
                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),

                    # Normalize
                    # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),     
                    ToTensord(keys=["n_20", "n_100"]),
                ])

    # Validation Augmentation
    else:
        if type == "full":
            return Compose([
                Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                AddChanneld(keys=["n_20", "n_100"]),     

                # Normalize
                # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                ToTensord(keys=["n_20", "n_100"])
            ])
        
        elif type == "window":
            return Compose([
                Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]
                AddChanneld(keys=["n_20", "n_100"]),     

                # Normalize
                # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                ToTensord(keys=["n_20", "n_100"]),
            ]) 

        elif type == "window_foreground":
            return Compose([
                Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]
                AddChanneld(keys=["n_20", "n_100"]),     

                CropForegroundd(keys=["n_20", "n_100"], source_key="n_100", select_fn=lambda x: x > 0),

                # Normalize
                ToTensord(keys=["n_20", "n_100"]),
            ])

 
# Sinogram Task
def Sinogram_Dataset_DCM(mode, type='window'):
    if mode == 'train':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/20/*/*/*.dcm'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/X/*/*/*.dcm'))
        files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]            
        transforms = get_transforms(mode='train', type=type)

        if type == 'full_patch' or type == 'window_patch':
            return Dataset(data=files, transform=transforms), list_data_collate
        else:
            return Dataset(data=files, transform=transforms), default_collate_fn        
            
    elif mode == 'valid':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/20/*/*/*.dcm'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/X/*/*/*.dcm'))
        files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]
        transforms = get_transforms(mode='valid', type=type)

        return Dataset(data=files, transform=transforms), default_collate_fn


# TEST Sinogram Task
def TEST_Sinogram_Dataset_DCM(mode, type):
    low_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/20/*/*/*.dcm'))
    high_imgs = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/X/*/*/*.dcm'))

    files = [{"n_20": low_name, "n_100": high_name, "path_n_20":low_path, "path_n_100":high_path} for low_name, high_name, low_path, high_path in zip(low_imgs, high_imgs, low_imgs, high_imgs)]
    transforms = get_transforms(mode=mode, type=type)

    return Dataset(data=files, transform=transforms), default_collate_fn

