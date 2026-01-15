import torch之后立即加
torch.cuda.device_count()
否则torch就查找不到cuda，很奇怪

export CUDA_VISIBLE_DEVICES=1

watch -n 1 nvidia-smi

nohup python -u test_pcc.py > log_base1 2>&1 &

nohup python -u train.py > log_base1 2>&1 &


pkill -9 python