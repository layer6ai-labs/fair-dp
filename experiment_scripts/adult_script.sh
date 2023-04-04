#!/bin/bash

for seed in {0..4}
do
  dir="adult_$seed"
  angles='False'  
  hessian='False'
  step=50

  echo "$seed nonpriv"
  python3 main.py --dataset=adult --method=regular --config group_ratios=1,1 --config make_valid_loader=0 --config net=mlp --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=20 --config evaluate_angles=$angles --config evaluate_hessian=$hessian --config angle_comp_step=$step --config logdir=$dir/adult_nonpriv --config seed=$seed

  echo "$seed dpsgd"
  python3 main.py --dataset=adult --method=dpsgd --config group_ratios=1,1 --config make_valid_loader=0 --config net=mlp --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=20 --config delta=1e-6 --config noise_multiplier=1 --config l2_norm_clip=0.5 --config evaluate_angles=$angles --config evaluate_hessian=$hessian --config angle_comp_step=$step --config logdir=$dir/adult_dpsgd --config seed=$seed

  echo "$seed dpsgd-f"
  python3 main.py --dataset=adult --method=dpsgd-f --config group_ratios=1,1 --config make_valid_loader=0 --config net=mlp --config lr=0.01 --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=20 --config delta=1e-6 --config noise_multiplier=1 --config base_max_grad_norm=0.5 --config counts_noise_multiplier=10 --config evaluate_angles=$angles --config evaluate_hessian=$hessian --config angle_comp_step=$step --config logdir=$dir/adult_dpsgdf --config seed=$seed

  echo "$seed dpsgd-g"
  python3 main.py --dataset=adult --method=dpsgd-global --config group_ratios=1,1 --config make_valid_loader=0 --config net=mlp --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=20 --config delta=1e-6 --config noise_multiplier=1 --config l2_norm_clip=0.5 --config evaluate_angles=$angles --config evaluate_hessian=$hessian --config angle_comp_step=$step  --config strict_max_grad_norm=50 --config lr=0.2 --config logdir=$dir/adult_dpsgdg --config seed=$seed

  echo "$seed dpsgd-g-adapt"
  python3 main.py --dataset=adult --method=dpsgd-global-adapt --config group_ratios=1,1 --config make_valid_loader=0 --config net=mlp --config train_batch_size=256 --config valid_batch_size=256 --config test_batch_size=256 --config max_epochs=20 --config delta=1e-6 --config noise_multiplier=1 --config l2_norm_clip=0.5 --config evaluate_angles=$angles --config evaluate_hessian=$hessian --config angle_comp_step=$step  --config strict_max_grad_norm=50 --config lr=0.2 --config bits_noise_multiplier=10 --config lr_Z=0.1 --config threshold=1 --config logdir=$dir/adult_dpsgdg --config seed=$seed
done
