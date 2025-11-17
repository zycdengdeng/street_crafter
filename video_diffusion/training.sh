##### debug
# CUDA_VISIBLE_DEVICES=0, python train.py \
#     --base configs/condition/waymo_low_res.yaml \
#     --num_nodes 1 \
#     --n_devices 1 \
#     --no_date True \
#     --finetune 'ckpts/vista.safetensors' \
#     --postfix '_debug' 
##### training

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --base configs/condition/waymo_low_res.yaml \
    --num_nodes 1 \
    --n_devices 8 \
    --finetune 'ckpts/vista.safetensors' \
    --no_date True

cd ./logs/condition-waymo_low_res/checkpoints/last.ckpt;

python zero_to_fp32.py . pytorch_model.bin;

cd /lpai/volumes/jointmodel/yanyunzhi/code/Vista;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --base configs/condition/waymo_high_res_mix.yaml \
    --num_nodes 1 \
    --n_devices 8 \
    --no_date True \
    --finetune ./logs/condition-waymo_low_res/checkpoints/last.ckpt/pytorch_model.bin