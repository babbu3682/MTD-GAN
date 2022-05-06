# FD-Net - Official Pytorch Implementation

It's scheduled to be uploaded soon. We are modifying the code for easier use.
We proposed a supervised multi-task aiding representation transfer learning network called <b>FD-Net</b>.


## 💡 Highlights
<!-- + Develop a robust feature extractor of brain hemorrhage in head & neck NCCT through three kinds of -->
<!-- multi-task representation learning. -->



<!-- <p align="center"><img width="100%" src="figures/graphical_abstract.png" /></p> -->


## Paper
This repository provides the official implementation of training SMART-Net as well as the usage of the pre-trained FD-Net in the following paper:

<!-- <b>Improved performance and robustness of multi-task representation learning with consistency loss between pretexts for intracranial hemorrhage identification in head CT</b> <br/>
[Sunggu Kyung](https://github.com/babbu3682)<sup>1</sup>, Keewon Shin, Hyunsu Jeong, Ki Duk Kim, Jooyoung Park, Kyungjin Cho, Jeong Hyun Lee, Gil-Sun Hong, and Namkug Kim <br/>
[MI2RL LAB](https://www.mi2rl.co/) <br/>
<b>(Under revision...)</b> Medical Image Analysis (MedIA) <br/>
<!-- [paper](https://arxiv.org/pdf/2004.07882.pdf) | [code](https://github.com/babbu3682/SMART-Net) | [graphical abstract](https://ars.els-cdn.com/content/image/1-s2.0-S1361841520302048-fx1_lrg.jpg) -->
[code](https://github.com/babbu3682/SMART-Net) -->


## Requirements
+ Linux
+ Python 3.8.5
+ PyTorch 1.8.0


## 📦 FD-Net Framework
### 1. Clone the repository and install dependencies
```bash
$ git clone https://github.com/babbu3682/FD-Net.git
$ cd FD-Net/
$ pip install -r requirements.txt
```

### 2. Preparing data
#### For your convenience, we have provided few 3D nii samples from [Physionet publish dataset](https://physionet.org/content/ct-ich/1.3.1/) as well as their mask labels. 
#### Note: We do not use this data as a train, it is just for code publishing examples.

<!-- Download the data from [this repository](https://zenodo.org/record/4625321/files/TransVW_data.zip?download=1).  -->
You can use your own data using the [dicom2nifti](https://github.com/icometrix/dicom2nifti) for converting from dicom to nii.

- The processed hemorrhage directory structure
```
datasets/samples/
    train
        |--  sample1_hemo_img.nii.gz
        |--  sample1_hemo_mask.nii
        |--  sample2_normal_img.nii.gz
        |--  sample2_normal_mask.nii        
                .
                .
                .
    valid
        |--  sample9_hemo_img.nii.gz
        |--  sample9_hemo_mask.nii
        |--  sample10_normal_img.nii.gz
        |--  sample10_normal_mask.nii
                .
                .
                .
    test
        |--  sample20_hemo_img.nii.gz
        |--  sample20_hemo_mask.nii
        |--  sample21_normal_img.nii.gz
        |--  sample21_normal_mask.nii
                .
                .
                .   
```

### 3. Upstream

#### 📋 Available List
- [x] Up_SMART_Net
- [x] Up_SMART_Net_Dual_CLS_SEG
- [x] Up_SMART_Net_Dual_CLS_REC
- [x] Up_SMART_Net_Dual_SEG_REC
- [x] Up_SMART_Net_Single_CLS
- [x] Up_SMART_Net_Single_SEG
- [x] Up_SMART_Net_Single_REC


**+ train**: We conducted upstream training with three multi-task including classificatiom, segmentation and reconstruction.
```bash
python train.py \
--data-folder-dir '/workspace/sunggu/4.Dose_img2img/datasets/[sinogram]Brain_3mm_DCM' \
--model-name 'Markovian_Patch_GAN' \
--lr 1e-4 \
--min-lr 1e-6 \
--batch-size 100 \
--epochs 500 \
--num_workers 8 \
--pin-mem \
--criterion 'L1 Loss' \
--criterion_mode 'not balance' \
--multiple_GT "False" \
--windowing "True" \
--patch_training "True" \
--multi-gpu-mode 'Single' \
--cuda-visible-devices '3' \
--print-freq 1 \
--save-checkpoint-every 2 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/model/[Previous]Markovian_Patch_GAN' \
--png-save-dir '/workspace/sunggu/4.Dose_img2img/Predictions/Train/[Previous]Markovian_Patch_GAN/'
```
<!-- **+ test**: We conducted upstream training with three multi-task including classificatiom, segmentation and reconstruction.
```bash
python test.py \
--data-folder-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples' \
--test-dataset-name 'Custom' \
--slice-wise-manner "True" \
--model-name 'Up_SMART_Net' \
--num-workers 4 \
--pin-mem \
--training-stream 'Upstream' \
--multi-gpu-mode 'Single' \
--cuda-visible-devices '2' \
--print-freq 1 \
--output-dir '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/up_test' \
--resume '/workspace/sunggu/1.Hemorrhage/SMART-Net/checkpoints/up_test/epoch_0_checkpoint.pth'
``` -->


## Fail training methods
### [SACNN](https://github.com/mmlipku/sacnn), [SA-SNMP](https://github.com/reach2sbera/ldct_nonlocal)
```
⏳ The official code was performed, but the results were very poor, and we requested from the authors, but it was unresponsive.
```


## Excuse
For personal information security reasons of medical data in Korea, our data cannot be disclosed.


## 📝 Citation
If you use this code for your research, please cite our papers:
```
⏳ It's scheduled to be uploaded soon.
```

## 🤝 Acknowledgement
We build SMART-Net framework by referring to the released code at [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) and [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI). 
This is a patent-pending technology.


### 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/babbu3682/SMART-Net/blob/main/LICENSE)