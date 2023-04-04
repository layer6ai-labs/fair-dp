#!/bin/bash

seed=0
dir="celeba/$seed"
angles="False"
hessians="False"

echo "$seed nonpriv"
python3 main.py --dataset=celeba --method=regular --config lr=0.01 --config max_epochs=60 --config logdir=$dir/celeba_nonpriv --config seed=$seed --config evaluate_angles=$angles --config evaluate_hessian=$hessians --config angle_comp_step=200

echo "$seed dpsgd"
python3 main.py --dataset=celeba --method=dpsgd --config lr=0.01 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config logdir=$dir/celeba_dpsgd --config seed=$seed --config evaluate_angles=$angles --config evaluate_hessian=$hessians --config angle_comp_step=200

echo "$seed dpsgd-f"
python3 main.py --dataset=celeba --method=dpsgd-f--config lr=0.01 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config base_max_grad_norm=1 --config counts_noise_multiplier=8 --config logdir=$dir/celeba_dpsgdf --config seed=$seed --config evaluate_angles=$angles --config evaluate_hessian=$hessians --config angle_comp_step=200

echo "$seed dpsgd-g"
python3 main.py --dataset=celeba --method=dpsgd-global --config lr=0.1 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config strict_max_grad_norm=100 --config logdir=$dir/celeba_dpsgdg --config seed=$seed --config evaluate_angles=$angles --config evaluate_hessian=$hessians --config angle_comp_step=200

echo "$seed dpsgd-global-adapt"
python3 main.py --dataset=celeba --method=dpsgd-global-adapt --config lr=0.1 --config max_epochs=60 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config strict_max_grad_norm=50 --config logdir=$dir/celeba_dpsgdg_adapt --config threshold=0.7 --config seed=$seed --config evaluate_angles=$angles --config evaluate_hessian=$hessians --config angle_comp_step=200

