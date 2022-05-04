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


######################################################                    collate_fn            ########################################################
def default_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



######################################################                    Sinogram Task                             ########################################################
from torchvision import transforms as vision_transforms



def Sinogram_Dataset_DCM(mode, patch_training):
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

def Sinogram_Dataset_DCM_Windowing(mode, patch_training):
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
                    ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]             
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
                AddChanneld(keys=["n_20", "n_100"]),     

                # Normalize
                # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                ToTensord(keys=["n_20", "n_100"]),
            ]
        )    


    return Dataset(data=files, transform=transforms), default_collate_fn

def Sinogram_Dataset_DCM_Multiple_GT(mode, patch_training):
    if mode == 'train':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/20/*/*/*.dcm')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/20/*/*/*.dcm'))
        n_40_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/40/*/*/*.dcm')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/40/*/*/*.dcm'))
        n_60_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/60/*/*/*.dcm')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/60/*/*/*.dcm'))
        n_80_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/80/*/*/*.dcm')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/80/*/*/*.dcm'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/X/*/*/*.dcm'))  + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/X/*/*/*.dcm'))

        files = [{"n_20": n_20, "n_40": n_40, "n_60": n_60, "n_80": n_80, "n_100": n_100} for n_20, n_40, n_60, n_80, n_100 in zip(n_20_imgs, n_40_imgs, n_60_imgs, n_80_imgs, n_100_imgs)]            
        print("Train [Total]  number = ", len(n_20_imgs))

        # CT에 맞는 Augmentation
        if patch_training:
            transforms = Compose(
                [
                    Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),                 

                    # Crop  
                    # RandWeightedCropd(keys=["image"], w_key=["image"], spatial_size=(512,512,1), num_samples=1),
                    RandSpatialCropSamplesd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], roi_size=(64, 64), num_samples=8, random_center=True, random_size=False, meta_keys=None, allow_missing_keys=False), 
                        # patch training, next(iter(loader)) output : list로 sample 만큼,,, 그 List 안에 (B, C, H, W)

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    # RandZoomd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, min_zoom=0.8, max_zoom=1.2, align_corners=None, keep_size=True, allow_missing_keys=False), # whole image 일때는 괜찮지만, Patch를 뜯을때 사용하면, 치명적이다...

                    # Normalize
                    # Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=functools.partial(minmax_normalize, option=False)),                 
                    ToTensord(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),
                ]
            )  

        else :
            transforms = Compose(
                [
                    Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),                 

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    # RandZoomd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, min_zoom=0.8, max_zoom=1.2, align_corners=None, keep_size=True, allow_missing_keys=False), # whole image 일때는 괜찮지만, Patch를 뜯을때 사용하면, 치명적이다...

                    # Normalize
                    # Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                    ToTensord(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),
                ]
            )              

    elif mode == 'valid':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/20/*/*/*.dcm'))
        n_40_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/40/*/*/*.dcm'))
        n_60_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/60/*/*/*.dcm'))
        n_80_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/80/*/*/*.dcm'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/X/*/*/*.dcm'))        

        files = [{"n_20": n_20, "n_40": n_40, "n_60": n_60, "n_80": n_80, "n_100": n_100} for n_20, n_40, n_60, n_80, n_100 in zip(n_20_imgs, n_40_imgs, n_60_imgs, n_80_imgs, n_100_imgs)]            
        print("Valid [Total]  number = ", len(n_20_imgs))

        # CT에 맞는 Augmentation
        transforms = Compose(
            [
                Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=get_pixels_hu),
                Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=dicom_normalize),
                AddChanneld(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),     

                # Normalize
                # Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=functools.partial(minmax_normalize, option=False)),                             
                ToTensord(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),
            ]
        )    
        

    return Dataset(data=files, transform=transforms), default_collate_fn






def Sinogram_Dataset_DCM_SACNN(mode, patch_training):
    if mode == 'train':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/20/')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/20/'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/X/'))  + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/X/'))

        first_list  = []
        middle_list = []
        last_list   = []
        target_list = []

        for scan_path, target_path in zip(n_20_imgs, n_100_imgs):
            input_list  = list_sort_nicely(glob.glob(scan_path.replace('[sinogram]', '*') + '/*/*/*.dcm'))
            gt_list     = list_sort_nicely(glob.glob(target_path.replace('[sinogram]', '*') + '/*/*/*.dcm'))
            
            assert len(input_list) == len(gt_list)
                
            # for i in range(0, len(input_list)//3):

            #     first_list.append(input_list[i*3+0])
            #     middle_list.append(input_list[i*3+1])
            #     last_list.append(input_list[i*3+2])
            #     target_list.append(gt_list[i*3+1])

            for i in range(len(input_list)):
                if i-1 == -1:
                    first_list.append(input_list[0])
                    middle_list.append(input_list[0])
                    last_list.append(input_list[1])
                    target_list.append(gt_list[0])                  

                elif i+1 == len(input_list):
                    first_list.append(input_list[-2])
                    middle_list.append(input_list[-1])
                    last_list.append(input_list[-1])
                    target_list.append(gt_list[-1])  
                    
                else :
                    first_list.append(input_list[i-1])
                    middle_list.append(input_list[i])
                    last_list.append(input_list[i+1])
                    target_list.append(gt_list[i])  

        files = [{"n_20_f": n_20_f, "n_20_m": n_20_m, "n_20_l": n_20_l, "n_100": n_100} for n_20_f, n_20_m, n_20_l, n_100 in zip(first_list, middle_list, last_list, target_list)]            
        print("Train [Total]  number = ", len(n_20_imgs))

        # CT에 맞는 Augmentation
        if patch_training:
            transforms = Compose(
                [  
                    Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"]),    

                    # Crop  
                    RandSpatialCropSamplesd(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], roi_size=(64, 64), num_samples=8, random_center=True, random_size=False, meta_keys=None, allow_missing_keys=False), 
                        # patch training, next(iter(loader)) output : list로 sample 만큼,,, 그 List 안에 (B, C, H, W)

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    # RandZoomd(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], prob=0.1, min_zoom=0.8, max_zoom=1.2, align_corners=None, keep_size=True, allow_missing_keys=False), # whole image 일때는 괜찮지만, Patch를 뜯을때 사용하면, 치명적이다...
                    
                    # Normalize
                    # Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=functools.partial(minmax_normalize, option=False)),                         
                    ToTensord(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"]),
                ]
            )  

        else :
            transforms = Compose(
                [
                    Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"]),                 

                    # (15 degree rotation, vertical & horizontal flip & scaling)
                    RandRotate90d(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], prob=0.1, spatial_axes=[0, 1], allow_missing_keys=False),
                    RandFlipd(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], prob=0.1, range_x=np.pi/12, range_y=np.pi/12, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    # RandZoomd(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], prob=0.1, min_zoom=0.8, max_zoom=1.2, align_corners=None, keep_size=True, allow_missing_keys=False), # whole image 일때는 괜찮지만, Patch를 뜯을때 사용하면, 치명적이다...

                    # Normalize
                    # Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=functools.partial(minmax_normalize, option=False)),                         
                    ToTensord(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"]),
                ]
            )              

    elif mode == 'valid':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/20/'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Sample/X/'))        

        first_list  = []
        middle_list = []
        last_list   = []
        target_list = []

        for scan_path, target_path in zip(n_20_imgs, n_100_imgs):
            input_list  = list_sort_nicely(glob.glob(scan_path.replace('[sinogram]', '*') + '/*/*/*.dcm'))
            gt_list     = list_sort_nicely(glob.glob(target_path.replace('[sinogram]', '*') + '/*/*/*.dcm'))
            
            assert len(input_list) == len(gt_list)
            
            # for i in range(0, len(input_list)//3):

            #     first_list.append(input_list[i*3+0])
            #     middle_list.append(input_list[i*3+1])
            #     last_list.append(input_list[i*3+2])
            #     target_list.append(gt_list[i*3+1])

            for i in range(len(input_list)):
                if i-1 == -1:
                    first_list.append(input_list[0])
                    middle_list.append(input_list[0])
                    last_list.append(input_list[1])
                    target_list.append(gt_list[0])                  

                elif i+1 == len(input_list):
                    first_list.append(input_list[-2])
                    middle_list.append(input_list[-1])
                    last_list.append(input_list[-1])
                    target_list.append(gt_list[-1])  
                    
                else :
                    first_list.append(input_list[i-1])
                    middle_list.append(input_list[i])
                    last_list.append(input_list[i+1])
                    target_list.append(gt_list[i])  
                

        files = [{"n_20_f": n_20_f, "n_20_m": n_20_m, "n_20_l": n_20_l, "n_100": n_100} for n_20_f, n_20_m, n_20_l, n_100 in zip(first_list, middle_list, last_list, target_list)]            
        print("Valid [Total]  number = ", len(n_20_imgs))

        # CT에 맞는 Augmentation
        transforms = Compose(
            [
                Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=get_pixels_hu),
                Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=dicom_normalize),
                AddChanneld(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"]),       

                # Normalize
                # Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=functools.partial(minmax_normalize, option=False)),                       
                ToTensord(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"]),
            ]
        )    
    
    else :
        print('Error...!')

    return Dataset(data=files, transform=transforms), default_collate_fn






######################################################                 TEST   Sinogram Task                             ########################################################

def TEST_Sinogram_Dataset_OLD(mode, range_minus1_plus1):
    if mode == 'sinogram':
        low_imgs      = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Test/*/20/*/*/*.npy'))
        high_imgs     = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Test/*/X/*/*/*.npy'))

        dcm_low_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/20/*/*/*.dcm'))
        dcm_high_imgs = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/X/*/*/*.dcm'))

        files = [{"low": low_name, "high": high_name, "dcm_low" : dcm_low, "dcm_high" : dcm_high} for low_name, high_name, dcm_low, dcm_high in zip(low_imgs, high_imgs, dcm_low_imgs, dcm_high_imgs)]
          
        print("TEST [Total]  number = ", len(low_imgs))

        if range_minus1_plus1:
            # CT에 맞는 Augmentation
            transforms = Compose(
                [
                    LoadNumpyd(keys=["low", "high"]),
                    AddChanneld(keys=["low", "high"]), 
                    ToTensord(keys=["low", "high"]),
                    
                    # Unet_with_perceptual Option
                    Lambdad(keys=["low", "high"], func=vision_transforms.Normalize(mean=(0.5), std=(0.5))),
                ]
            )            
        else:
            # CT에 맞는 Augmentation
            transforms = Compose(
                [
                    LoadNumpyd(keys=["low", "high"]),
                    AddChanneld(keys=["low", "high"]), 
                    ToTensord(keys=["low", "high"]),
                ]
            )            

    # follow dataset 미완성...
    elif mode == 'follow':
        low_imgs      = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_Low_Dose_CT_Grand_Challenge_dataset_3mm/Test/*/20/*/*/*.npy'))
        high_imgs     = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_Low_Dose_CT_Grand_Challenge_dataset_3mm/Test/*/X/*/*/*.npy'))

        dcm_low_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/20/*/*/*.dcm'))
        dcm_high_imgs = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/X/*/*/*.dcm'))

        files = [{"low": low_name, "high": high_name, "dcm_low" : dcm_low, "dcm_high" : dcm_high} for low_name, high_name, dcm_low, dcm_high in zip(low_imgs, high_imgs, dcm_low_imgs, dcm_high_imgs)]
          
        print("TEST [Total]  number = ", len(low_imgs))

        # CT에 맞는 Augmentation
        transforms = Compose(
            [
                LoadNumpyd(keys=["low", "high"]),
                AddChanneld(keys=["low", "high"]), 
                ToTensord(keys=["low", "high"]),
                
                # Unet_with_perceptual Option
                Lambdad(keys=["low", "high"], func=vision_transforms.Normalize(mean=(0.5), std=(0.5))),
            ]
        )


    return Dataset(data=files, transform=transforms), default_collate_fn
    
def TEST_Sinogram_Dataset_DCM():
    low_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/20/*/*/*.dcm'))
    high_imgs = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/X/*/*/*.dcm'))

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

def TEST_Sinogram_Dataset_DCM_Windowing():
    low_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/20/*/*/*.dcm'))
    high_imgs = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/X/*/*/*.dcm'))

    files = [{"n_20": low_name, "n_100": high_name, "path_n_20":low_path, "path_n_100":high_path} for low_name, high_name, low_path, high_path in zip(low_imgs, high_imgs, low_imgs, high_imgs)]
    print("TEST [Total]  number = ", len(low_imgs))

    transforms = Compose(
        [
            Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
            ScaleIntensityRanged(keys=["n_20", "n_100"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),     # Windowing HU [min:0, max:80]             
            AddChanneld(keys=["n_20", "n_100"]),         

            # Normalize
            # Lambdad(keys=["n_20", "n_100"], func=functools.partial(minmax_normalize, option=False)),                         
            ToTensord(keys=["n_20", "n_100"]),
        ]
    )        

    return Dataset(data=files, transform=transforms), default_collate_fn

def TEST_Sinogram_Dataset_DCM_SACNN():
    low_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/20/'))
    high_imgs = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/X/'))

    first_list  = []
    middle_list = []
    last_list   = []
    target_list = []

    for scan_path, target_path in zip(low_imgs, high_imgs):
        input_list  = list_sort_nicely(glob.glob(scan_path.replace('[sinogram]', '*') + '/*/*/*.dcm'))
        gt_list     = list_sort_nicely(glob.glob(target_path.replace('[sinogram]', '*') + '/*/*/*.dcm'))
        
        assert len(input_list) == len(gt_list)
        
        # for i in range(0, len(input_list)//3):

        #     first_list.append(input_list[i*3+0])
        #     middle_list.append(input_list[i*3+1])
        #     last_list.append(input_list[i*3+2])
        #     target_list.append(gt_list[i*3+1])

        for i in range(len(input_list)):
            if i-1 == -1:
                first_list.append(input_list[0])
                middle_list.append(input_list[0])
                last_list.append(input_list[1])
                target_list.append(gt_list[0])                  

            elif i+1 == len(input_list):
                first_list.append(input_list[-2])
                middle_list.append(input_list[-1])
                last_list.append(input_list[-1])
                target_list.append(gt_list[-1])  

            else :
                first_list.append(input_list[i-1])
                middle_list.append(input_list[i])
                last_list.append(input_list[i+1])
                target_list.append(gt_list[i])  
            

    files = [{"n_20_f": n_20_f, "n_20_m": n_20_m, "n_20_l": n_20_l, "n_100": n_100, "path_n_20":n_20_m, "path_n_100":n_100} for n_20_f, n_20_m, n_20_l, n_100 in zip(first_list, middle_list, last_list, target_list)]            
    print("Valid [Total]  number = ", len(low_imgs))

    # CT에 맞는 Augmentation
    transforms = Compose(
        [
            Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=get_pixels_hu),
            Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=dicom_normalize),
            AddChanneld(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"]),       

            # Normalize
            # Lambdad(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"], func=functools.partial(minmax_normalize, option=False)),                   
            ToTensord(keys=["n_20_f", "n_20_m", "n_20_l", "n_100"]),
        ]
    )        

    return Dataset(data=files, transform=transforms), default_collate_fn