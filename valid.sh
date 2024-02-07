#!/bin/sh

############################### Abdomen ###############################

# abdomen Ablation CLS
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_CLS' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, ablation CLS, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation SEG
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_SEG' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_SEG' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_SEG' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_SEG/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, ablation CLS, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation CLS+SEG
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_CLS_SEG' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS_SEG' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, ablation CLS, 200 epoch, node g' \
    --epoch $epoch
done
END



# abdomen Ablation CLS+REC
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_CLS_REC' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_REC' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS_REC' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_REC/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, ablation CLS, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation SEG+REC
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_SEG_REC' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_SEG_REC' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_SEG_REC' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_SEG_REC/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, Ablation_SEG_REC, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation CLS+SEG+REC
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_CLS_SEG_REC' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS_SEG_REC' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, Ablation_CLS_SEG_REC, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation CLS+SEG+REC+NDS
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_CLS_SEG_REC_NDS' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_NDS' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS_SEG_REC_NDS' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_NDS/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, Ablation_CLS_SEG_REC_NDS, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation CLS+SEG+REC+RC
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_CLS_SEG_REC_RC' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_RC' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS_SEG_REC_RC' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_RC/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, Ablation_CLS_SEG_REC_RC, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation CLS+SEG+REC+RC+NDS
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_CLS_SEG_REC_NDS_RC' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_NDS_RC' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS_SEG_REC_NDS_RC' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_NDS_RC/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, Ablation_CLS_SEG_REC_NDS_RC, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation CLS+SEG+REC+RC+NDS+FFTGAN
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'MTD_GAN' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_NDS_RC_FFTGAN' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS_SEG_REC_NDS_RC_FFTGAN' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_NDS_RC_FFTGAN/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, Ablation_CLS_SEG_REC_NDS_RC_FFTGAN, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation CLS+SEG+REC+RC+NDS+FFTGAN+PCGrad
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'MTD_GAN' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_NDS_RC_FFTGAN_PCGrad' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS_SEG_REC_NDS_RC_FFTGAN_PCGrad' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_NDS_RC_FFTGAN_PCGrad/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, Ablation_CLS_SEG_REC_NDS_RC_FFTGAN_PCGrad, 200 epoch, node g' \
    --epoch $epoch
done
END


##########################################################
# abdomen Ablation CLS, ablation 2
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_CLS_None_SN' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_None_SN' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS_None_SN' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_None_SN/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, Ablation_CLS_None_SN, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation SEG, ablation 2
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'Ablation_SEG_None_SN' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_SEG_None_SN' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_SEG_None_SN' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_SEG_None_SN/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, Ablation_SEG_None_SN, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation CLS+SEG+REC+RC+NDS+FFTGAN+PCGrad, ablation 2
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'MTD_GAN_None_SN' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_NDS_RC_FFTGAN_PCGrad_None_SN' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/Ablation_CLS_SEG_REC_NDS_RC_FFTGAN_PCGrad_None_SN' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/Ablation_CLS_SEG_REC_NDS_RC_FFTGAN_PCGrad_None_SN/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, Ablation_CLS_SEG_REC_NDS_RC_FFTGAN_PCGrad_None_SN, 200 epoch, node g' \
    --epoch $epoch
done
END




##########################################################
# abdomen Ablation MTD_GAN_All_One, ablation 3
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'MTD_GAN_All_One' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/MTD_GAN_All_One' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/MTD_GAN_All_One' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/MTD_GAN_All_One/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, MTD_GAN_All_One, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation MTD_GAN_Manual, ablation 3
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'MTD_GAN_Manual' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/MTD_GAN_Manual' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/MTD_GAN_Manual' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/MTD_GAN_Manual/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, MTD_GAN_Manual, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation MTD_GAN_Method, ablation 3
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'MTD_GAN_Method' \
    --method 'mgda' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/MTD_GAN_Method_MGDA' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/MTD_GAN_Method_MGDA' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/MTD_GAN_Method_MGDA/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, MTD_GAN_Method_MGDA, 200 epoch, node g' \
    --epoch $epoch
done
END


# abdomen Ablation MTD_GAN_Method, ablation 3
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'MTD_GAN_Method' \
    --method 'pcgrad' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/MTD_GAN_Method_PCGrad' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/MTD_GAN_Method_PCGrad' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/MTD_GAN_Method_PCGrad/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, MTD_GAN_Method_PCGrad, 200 epoch, node g' \
    --epoch $epoch
done
END

# abdomen Ablation MTD_GAN_Method, ablation 3
: <<'END'
for epoch in {0..200}
do
    CUDA_VISIBLE_DEVICES=1 python -W ignore test.py \
    --dataset 'mayo' \
    --dataset-type-train 'window_patch' \
    --dataset-type-valid 'window' \
    --batch-size 1 \
    --train-num-workers 1 \
    --valid-num-workers 4 \
    --model 'MTD_GAN_Method' \
    --method 'cagrad' \
    --loss 'L1 Loss' \
    --multi-gpu-mode 'Single' \
    --device 'cuda' \
    --print-freq 10 \
    --checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/MTD_GAN_Method_CAGrad' \
    --save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/MTD_GAN_Method_CAGrad' \
    --resume '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/MTD_GAN_Method_CAGrad/epoch_'${epoch}'_checkpoint.pth' \
    --memo 'abdomen, MTD_GAN_Method_CAGrad, 200 epoch, node g' \
    --epoch $epoch
done
END


## abdomen Previosue
# RED_CNN
: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
--dataset 'mayo' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 100 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'RED_CNN' \
--loss 'L1 Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/RED_CNN' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/RED_CNN' \
--memo 'abdomen, RED_CNN, 500 epoch, node g'
END


# DDPM
: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
--dataset 'mayo' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 20 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'DDPM' \
--loss 'L1 Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/abdomen/DDPM' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/abdomen/DDPM' \
--memo 'abdomen, DDPM, 500 epoch, node g'
END


############################### Brain ###############################

# brain Ablation CLS
: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 30 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'Ablation_CLS' \
--loss 'L1 Loss' \
--pcgrad 'False' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/Ablation_CLS' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/Ablation_CLS' \
--memo 'brain, ablation CLS, 200 epoch, node g'
END

# brain Ablation SEG
: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 30 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'Ablation_SEG' \
--loss 'L1 Loss' \
--pcgrad 'False' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/Ablation_SEG' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/Ablation_SEG' \
--memo 'brain, ablation SEG, 200 epoch, node g'
END 

# brain Ablation CLS+SEG
: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 30 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'Ablation_CLS_SEG' \
--loss 'L1 Loss' \
--pcgrad 'False' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/Ablation_CLS_SEG' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/Ablation_CLS_SEG' \
--memo 'brain, ablation SEG, 200 epoch, node g'
END

# brain Ablation CLS+REC
: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 30 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'Ablation_CLS_REC' \
--loss 'L1 Loss' \
--pcgrad 'False' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/Ablation_CLS_REC' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/Ablation_CLS_REC' \
--memo 'brain, ablation SEG, 200 epoch, node g'
END

# brain Ablation SEG+REC
: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 30 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'Ablation_SEG_REC' \
--loss 'L1 Loss' \
--pcgrad 'False' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/Ablation_SEG_REC' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/Ablation_SEG_REC' \
--memo 'brain, ablation SEG, 200 epoch, node g'
END

# brain Ablation CLS+SEG+REC
: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 30 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'Ablation_CLS_SEG_REC' \
--loss 'L1 Loss' \
--pcgrad 'False' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/Ablation_CLS_SEG_REC' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/Ablation_CLS_SEG_REC' \
--memo 'brain, ablation SEG, 200 epoch, node g'
END

# brain Ablation CLS+SEG+REC+NDS
: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 30 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'Ablation_CLS_SEG_REC_NDS' \
--loss 'L1 Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/Ablation_CLS_SEG_REC_NDS' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/Ablation_CLS_SEG_REC_NDS' \
--memo 'brain, ablation SEG, 200 epoch, node g'
END

# brain Ablation CLS+SEG+REC+RC
: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 30 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'Ablation_CLS_SEG_REC_RC' \
--loss 'L1 Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/Ablation_CLS_SEG_REC_RC' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/Ablation_CLS_SEG_REC_RC' \
--memo 'brain, ablation SEG, 200 epoch, node g'
END

# brain Ablation CLS+SEG+REC+RC+NDS
: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 30 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'Ablation_CLS_SEG_REC_NDS_RC' \
--loss 'L1 Loss' \
--pcgrad 'False' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/Ablation_CLS_SEG_REC_NDS_RC' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/Ablation_CLS_SEG_REC_NDS_RC' \
--memo 'brain, ablation SEG, 200 epoch, node g'
END

# brain Ablation CLS+SEG+REC+RC+NDS+PcGrad
: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 30 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'Ablation_CLS_SEG_REC_NDS_RC' \
--loss 'L1 Loss' \
--pcgrad 'True' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/Ablation_CLS_SEG_REC_NDS_RC' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/Ablation_CLS_SEG_REC_NDS_RC' \
--memo 'brain, ablation SEG, 200 epoch, node g'
END



# RED_CNN
: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
--dataset 'amc' \
--dataset-type-train 'window_patch' \
--dataset-type-valid 'window' \
--batch-size 100 \
--train-num-workers 16 \
--valid-num-workers 4 \
--model 'RED_CNN' \
--loss 'L1 Loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 200 \
--warmup-epochs 10 \
--lr 1e-4 \
--min-lr 1e-6 \
--multi-gpu-mode 'Single' \
--device 'cuda' \
--print-freq 10 \
--save-checkpoint-every 1 \
--checkpoint-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/checkpoints/brain/RED_CNN' \
--save-dir '/workspace/sunggu/4.Dose_img2img/MTD_GAN/predictions/train/brain/RED_CNN' \
--memo 'brain, RED_CNN, 200 epoch, node g'
END



