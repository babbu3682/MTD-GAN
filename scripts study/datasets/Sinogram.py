import torch
import numpy as np
from monai.transforms import *
from monai.data import Dataset
import re
import glob
import pydicom

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



######################################################                    collate_fn            ########################################################
def default_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)



######################################################                    Sinogram Task                             ########################################################
from torchvision import transforms as vision_transforms

def Sinogram_Dataset_NPY(mode, patch_training, multiple_GT):
    if mode == 'train':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Train/*/20/*/*/*.npy')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Valid/*/20/*/*/*.npy'))
        n_40_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Train/*/40/*/*/*.npy')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Valid/*/40/*/*/*.npy'))
        n_60_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Train/*/60/*/*/*.npy')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Valid/*/60/*/*/*.npy'))
        n_80_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Train/*/80/*/*/*.npy')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Valid/*/80/*/*/*.npy'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Train/*/X/*/*/*.npy'))  + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Valid/*/X/*/*/*.npy'))

        files = [{"n_20": n_20, "n_40": n_40, "n_60": n_60, "n_80": n_80, "n_100": n_100} for n_20, n_40, n_60, n_80, n_100 in zip(n_20_imgs, n_40_imgs, n_60_imgs, n_80_imgs, n_100_imgs)]
          
        print("Train [Total]  number = ", len(n_20_imgs))

        # CT에 맞는 Augmentation
        transforms = Compose(
            [
                LoadNumpyd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),
                AddChanneld(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),                 
                ToTensord(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),
            ]
        )    

    elif mode == 'valid':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Valid/*/20/*/*/*.npy'))
        n_40_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Valid/*/40/*/*/*.npy'))
        n_60_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Valid/*/60/*/*/*.npy'))
        n_80_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Valid/*/80/*/*/*.npy'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Valid/*/X/*/*/*.npy'))

        files = [{"n_20": n_20, "n_40": n_40, "n_60": n_60, "n_80": n_80, "n_100": n_100} for n_20, n_40, n_60, n_80, n_100 in zip(n_20_imgs, n_40_imgs, n_60_imgs, n_80_imgs, n_100_imgs)]
          
        print("Valid [Total]  number = ", len(n_20_imgs))

        # CT에 맞는 Augmentation
        transforms = Compose(
            [
                LoadNumpyd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),
                AddChanneld(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),                 
                ToTensord(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),
            ]
        )    


    else :
        low_imgs      = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Test/*/20/*/*/*.npy'))
        high_imgs     = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NPY/Test/*/X/*/*/*.npy'))

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

def Sinogram_Dataset_DCM(mode, patch_training, multiple_GT):
    if multiple_GT:
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
                        # RandSpatialCropd(keys=["image"], roi_size=(512, 512), random_size=False, random_center=True),
                        # RandSpatialCropd(keys=["image"], roi_size=(512,512,3), random_size=False, random_center=True),
                        RandSpatialCropSamplesd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], roi_size=(64, 64), num_samples=8, random_center=True, random_size=False, meta_keys=None, allow_missing_keys=False), 
                            # patch training, next(iter(loader)) output : list로 sample 만큼,,, 그 List 안에 (B, C, H, W)

                        # (45 degree rotation, vertical & horizontal flip & scaling)
                        RandFlipd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                        RandRotated(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                        RandZoomd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),
                        ToTensord(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),
                    ]
                )  

            else :
                transforms = Compose(
                    [
                        Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=get_pixels_hu),
                        Lambdad(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], func=dicom_normalize),
                        AddChanneld(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),                 

                        # (45 degree rotation, vertical & horizontal flip & scaling)
                        RandFlipd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                        RandRotated(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                        RandZoomd(keys=["n_20", "n_40", "n_60", "n_80", "n_100"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),
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
                    ToTensord(keys=["n_20", "n_40", "n_60", "n_80", "n_100"]),
                ]
            )    
        
        else :
            print('Error...!')

    else:
        if mode == 'train':
            n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/20/*/*/*.dcm')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/20/*/*/*.dcm'))
            n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Train/*/X/*/*/*.dcm'))  + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Valid/*/X/*/*/*.dcm'))

            files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]            
            print("Train [Total]  number = ", len(n_20_imgs))

            # CT에 맞는 Augmentation
            if patch_training:
                transforms = Compose(
                    [
                        Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                        Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                        AddChanneld(keys=["n_20", "n_100"]),                 

                        # Crop  
                        RandSpatialCropSamplesd(keys=["n_20", "n_100"], roi_size=(64, 64), num_samples=8, random_center=True, random_size=False, meta_keys=None, allow_missing_keys=False), 
                            # patch training, next(iter(loader)) output : list로 sample 만큼,,, 그 List 안에 (B, C, H, W)

                        # (45 degree rotation, vertical & horizontal flip & scaling)
                        RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                        RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                        RandZoomd(keys=["n_20", "n_100"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),
                        ToTensord(keys=["n_20", "n_100"]),
                    ]
                )  

            else :
                transforms = Compose(
                    [
                        Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                        Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                        AddChanneld(keys=["n_20", "n_100"]),                 

                        # (45 degree rotation, vertical & horizontal flip & scaling)
                        RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                        RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                        RandZoomd(keys=["n_20", "n_100"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),
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
                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )    
        
        else :
            print('Error...!')

    return Dataset(data=files, transform=transforms), default_collate_fn

def Sinogram_Dataset_NII(mode, patch_training, multiple_GT):
    if mode == 'train':
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NII/Train/*/20/*/*/*.nii.gz')) + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NII/Valid/*/20/*/*/*.nii.gz'))
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NII/Train/*/X/*/*/*.nii.gz'))  + list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NII/Valid/*/X/*/*/*.nii.gz'))

        files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]
          
        print("Train [Total]  number = ", len(n_20_imgs))

        # CT에 맞는 Augmentation
        if patch_training:
            transforms = Compose(
                [
                    LoadImaged(keys=["n_20", "n_100"]),
                    AddChanneld(keys=["n_20", "n_100"]),                 
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),       # 보통 먼저 normalize 하고 aug 하는 경우가 있음 ref: REDCNN

                    # Align
                    Flipd(keys=["n_20", "n_100"], spatial_axis=1),
                    Rotate90d(keys=["n_20", "n_100"], k=1, spatial_axes=(0, 1)),          

                    # Crop  
                    # RandWeightedCropd(keys=["image"], w_key=["image"], spatial_size=(512,512,1), num_samples=1),
                    # RandSpatialCropd(keys=["image"], roi_size=(512, 512), random_size=False, random_center=True),
                    # RandSpatialCropd(keys=["image"], roi_size=(512,512,3), random_size=False, random_center=True),

                    RandSpatialCropSamplesd(keys=["n_20", "n_100"], roi_size=(32, 32, 3), num_samples=8, random_center=True, random_size=False, meta_keys=None, allow_missing_keys=False), 
                    # for SACNN num_samples down to 4....
                    # RandSpatialCropSamplesd(keys=["n_20", "n_100"], roi_size=(64, 64, 3), num_samples=2, random_center=True, random_size=False, meta_keys=None, allow_missing_keys=False), 
                        # patch training, next(iter(loader)) output : list로 sample 만큼,,, 그 List 안에 (B, C, H, W)

                    # (45 degree rotation, vertical & horizontal flip & scaling)
                    RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    RandZoomd(keys=["n_20", "n_100"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),

                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )    
        else :
            transforms = Compose(
                [
                    LoadImaged(keys=["n_20", "n_100"]),
                    AddChanneld(keys=["n_20", "n_100"]),                 
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),       # 보통 먼저 normalize 하고 aug 하는 경우가 있음 ref: REDCNN             

                    # Align
                    Flipd(keys=["n_20", "n_100"], spatial_axis=1),
                    Rotate90d(keys=["n_20", "n_100"], k=1, spatial_axes=(0, 1)),  

                    # (45 degree rotation, vertical & horizontal flip & scaling)
                    RandFlipd(keys=["n_20", "n_100"], prob=0.1, spatial_axis=[0, 1], allow_missing_keys=False),
                    RandRotated(keys=["n_20", "n_100"], prob=0.1, range_x=np.pi/4, range_y=np.pi/4, range_z=0.0, keep_size=True, align_corners=False, allow_missing_keys=False),
                    RandZoomd(keys=["n_20", "n_100"], prob=0.1, min_zoom=0.5, max_zoom=2.0, align_corners=None, keep_size=True, allow_missing_keys=False),

                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )              

    elif mode == 'valid':
        # n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NII/Valid/*/20/*/*/*.nii.gz'))
        # n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NII/Valid/*/X/*/*/*.nii.gz'))
        n_20_imgs   = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NII/Sample/20/*/*/*.nii.gz'))        
        n_100_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NII/Sample/X/*/*/*.nii.gz'))        

        files = [{"n_20": n_20, "n_100": n_100} for n_20, n_100 in zip(n_20_imgs, n_100_imgs)]
          
        print("Valid [Total]  number = ", len(n_20_imgs))

        # CT에 맞는 Augmentation
        transforms = Compose(
            [
                LoadImaged(keys=["n_20", "n_100"]),
                AddChanneld(keys=["n_20", "n_100"]),                 
                Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),       # 보통 먼저 normalize 하고 aug 하는 경우가 있음 ref: REDCNN             

                # Align
                Flipd(keys=["n_20", "n_100"], spatial_axis=1),
                Rotate90d(keys=["n_20", "n_100"], k=1, spatial_axes=(0, 1)),  

                ToTensord(keys=["n_20", "n_100"]),
            ]
        )    
    
    else: 
        raise Exception('Error...! mode')

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
    
def TEST_Sinogram_Dataset_DCM(mode, range_minus1_plus1):
    if mode == 'sinogram':
        low_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/20/*/*/*.dcm'))
        high_imgs = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_DCM/Test/*/X/*/*/*.dcm'))

        files = [{"n_20": low_name, "n_100": high_name, "path_n_20":low_path, "path_n_100":high_path} for low_name, high_name, low_path, high_path in zip(low_imgs, high_imgs, low_imgs, high_imgs)]
          
        print("TEST [Total]  number = ", len(low_imgs))

        if range_minus1_plus1:
            transforms = Compose(
                [
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20", "n_100"]),                 
                    ToTensord(keys=["n_20", "n_100"]),
                    # Unet_with_perceptual Option
                    Lambdad(keys=["n_20", "n_100"], func=vision_transforms.Normalize(mean=(0.5), std=(0.5))),
                ]
            )            
        else:
            transforms = Compose(
                [
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20", "n_100"]),                 
                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )        

    # follow dataset 미완성...
    elif mode == 'follow':
        low_imgs      = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_Low_Dose_CT_Grand_Challenge_dataset_3mm/Test/*/20/*/*/*.dcm'))
        high_imgs     = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_Low_Dose_CT_Grand_Challenge_dataset_3mm/Test/*/X/*/*/*.dcm'))

        files = [{"n_20": low_name, "n_100": high_name} for low_name, high_name in zip(low_imgs, high_imgs)]
          
        print("TEST [Total]  number = ", len(low_imgs))

        if range_minus1_plus1:
            transforms = Compose(
                [
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20", "n_100"]),                 
                    ToTensord(keys=["n_20", "n_100"]),
                    # Unet_with_perceptual Option
                    Lambdad(keys=["n_20", "n_100"], func=vision_transforms.Normalize(mean=(0.5), std=(0.5))),
                ]
            )            
        else:
            transforms = Compose(
                [
                    Lambdad(keys=["n_20", "n_100"], func=get_pixels_hu),
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),
                    AddChanneld(keys=["n_20", "n_100"]),                 
                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )  


    return Dataset(data=files, transform=transforms), default_collate_fn

def TEST_Sinogram_Dataset_NII(mode, range_minus1_plus1):
    if mode == 'sinogram':
        
        low_imgs  = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NII/Test/*/20/*/*/*.nii.gz'))
        high_imgs = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_3mm_NII/Test/*/X/*/*/*.nii.gz'))

        files = [{"n_20": low_name, "n_100": high_name} for low_name, high_name in zip(low_imgs, high_imgs)]
          
        print("TEST [Total]  number = ", len(low_imgs))

        if range_minus1_plus1:
            transforms = Compose(
                [
                    LoadImaged(keys=["n_20", "n_100"]),
                    AddChanneld(keys=["n_20", "n_100"]),                 
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),       # 보통 먼저 normalize 하고 aug 하는 경우가 있음 ref: REDCNN             

                    # Align
                    Flipd(keys=["n_20", "n_100"], spatial_axis=1),
                    Rotate90d(keys=["n_20", "n_100"], k=1, spatial_axes=(0, 1)),  

                    ToTensord(keys=["n_20", "n_100"]),

                    # Unet_with_perceptual Option
                    Lambdad(keys=["n_20", "n_100"], func=vision_transforms.Normalize(mean=(0.5), std=(0.5))),
                ]
            )            
        else:            
            transforms = Compose(
                [
                    LoadImaged(keys=["n_20", "n_100"]),
                    AddChanneld(keys=["n_20", "n_100"]),                 
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),       # 보통 먼저 normalize 하고 aug 하는 경우가 있음 ref: REDCNN             

                    # Align
                    Flipd(keys=["n_20", "n_100"], spatial_axis=1),
                    Rotate90d(keys=["n_20", "n_100"], k=1, spatial_axes=(0, 1)),  

                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )    

    # follow dataset 미완성...
    elif mode == 'follow':
        low_imgs      = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_Low_Dose_CT_Grand_Challenge_dataset_3mm/Test/*/20/*/*/*.dcm'))
        high_imgs     = list_sort_nicely(glob.glob('/workspace/sunggu/4.Dose_img2img/dataset/*Brain_Low_Dose_CT_Grand_Challenge_dataset_3mm/Test/*/X/*/*/*.dcm'))

        files = [{"n_20": low_name, "n_100": high_name} for low_name, high_name in zip(low_imgs, high_imgs)]
          
        print("TEST [Total]  number = ", len(low_imgs))

        if range_minus1_plus1:
            transforms = Compose(
                [
                    LoadImaged(keys=["n_20", "n_100"]),
                    AddChanneld(keys=["n_20", "n_100"]),                 
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),       # 보통 먼저 normalize 하고 aug 하는 경우가 있음 ref: REDCNN             

                    # Align
                    Flipd(keys=["n_20", "n_100"], spatial_axis=1),
                    Rotate90d(keys=["n_20", "n_100"], k=1, spatial_axes=(0, 1)),  

                    ToTensord(keys=["n_20", "n_100"]),

                    # Unet_with_perceptual Option
                    Lambdad(keys=["n_20", "n_100"], func=vision_transforms.Normalize(mean=(0.5), std=(0.5))),
                ]
            )            
        else:            
            transforms = Compose(
                [
                    LoadImaged(keys=["n_20", "n_100"]),
                    AddChanneld(keys=["n_20", "n_100"]),                 
                    Lambdad(keys=["n_20", "n_100"], func=dicom_normalize),       # 보통 먼저 normalize 하고 aug 하는 경우가 있음 ref: REDCNN             

                    # Align
                    Flipd(keys=["n_20", "n_100"], spatial_axis=1),
                    Rotate90d(keys=["n_20", "n_100"], k=1, spatial_axes=(0, 1)),  

                    ToTensord(keys=["n_20", "n_100"]),
                ]
            )    


    return Dataset(data=files, transform=transforms), default_collate_fn