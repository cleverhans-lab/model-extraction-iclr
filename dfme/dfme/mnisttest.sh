#!/bin/bash

#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --partition=t4v2

source ~/envnew
python train.py --dataset mnist --device 0 --grad_m 1 --query_budget 2 --log_dir save_results/mnistnewpate --lr_G 5e-5 --student_model resnet18_8x --loss l1 --steps 0.5 0.8