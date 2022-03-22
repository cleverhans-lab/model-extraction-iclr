#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
python train.py --dataset cifar10 --ckpt checkpoint/teacher/cifar10-resnet34_8x.pt --device 0 --grad_m 1 --query_budget 20 --log_dir save_results/cifar10new  --lr_G 1e-4 --student_model resnet18_8x --model resnet34_8x --loss l1