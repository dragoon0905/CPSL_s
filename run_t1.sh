#!/bin/bash

#SBATCH --job-name=CPSL
#SBATCH --gres=gpu:4
#SBATCH -o ./train_out/train.out
#SBATCH -w agi1
#SBATCH --mem-per-gpu=10G
#SBATCH -p batch
#SBATCH --cpus-per-task=4
#SBATCH --time=14-0

source /data/seunan/init.sh
conda activate torch38gpu




# stage1 
#generate soft pseudo label
#python generate_pseudo_label.py --name gta2citylabv2_warmup_soft --soft --resume_path  \
#./pretrained_models/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast

#Calculate prototypes for weight initialization
#python calc_prototype.py --resume_path \
#./pretrained_models/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl

#Calculate class disrtibution.
#python generate_class_distribution.py --name gta2citylabv2_warmup_soft --soft \
#--resume_path  ./pretrained_models/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast --class_balance

# train the model
python train.py --name gta2citylabv2_stage1Denoise --used_save_pseudo --ema --proto_rectify \
--path_soft Pseudo/gta2citylabv2_warmup_soft \
--resume_path ./pretrained_models/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl \
--rce --proto_consistW 5 --SL_lambda 0.1


#stage 2
#generate soft pseudo label
#python generate_pseudo_label.py --name gta2citylabv2_stage1Denoise --flip \
#--resume_path ./logs/gta2citylabv2_stage1Denoise/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast

#python train.py --name gta2citylabv2_stage2 --stage stage2 --used_save_pseudo \
#--path_LP Pseudo/gta2citylabv2_stage1Denoise \
#--resume_path ./logs/gta2citylabv2_stage1Denoise/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl \
#--S_pseudo 1 --threshold 0.95 --distillation 1 --finetune --lr 6e-4 --student_init simclr --bn_clr --no_resume

#stage 3
#generate soft pseudo label
#python generate_pseudo_label.py --name gta2citylabv2_stage2 --flip \
#--resume_path ./logs/gta2citylabv2_stage2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast\
#--bn_clr --student_init simclr

#python train.py --name gta2citylabv2_stage3 --stage stage3 --used_save_pseudo \
#--path_LP Pseudo/gta2citylabv2_stage2 \
#--resume_path ./logs/gta2citylabv2_stage2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl \
#--S_pseudo 1 --threshold 0.95 --distillation 1 --finetune --lr 6e-4 --student_init simclr --bn_clr --ema_bn