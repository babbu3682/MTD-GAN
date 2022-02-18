## Train code

python -m torch.distributed.launch \
--nproc_per_node=4 \
--use_env main.py \
--batch-size 170 \
--epochs 501 \
--min-lr 5e-6 \
--lr 1e-3 \
--training-mode 'SSL' \
--data-set 'PedXnet' \
--output 'checkpoints/SSL/PedXnet' \
--validate-every 1 \
--resume '/workspace/sunggu/3.Child/SiT-main/checkpoints/SSL/PedXnet/checkpoint.pth' \
--num_workers 32 \
--gpus "0, 3, 5, 6"