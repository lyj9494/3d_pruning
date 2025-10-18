# 
# **Updated on 10.18 16:38**

## The function of the current version 
(1) train a block- and timestep-wise mask for 3d generation method triposg (using the l2c baseline method)
```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 CUDA_VISIBLE_DEVICES=3,4,5,6,7  \
    accelerate launch --multi_gpu --num_processes 5   \
        TripoSG/scripts/train/train_triposg_mask.py  \
            --config configs/mp8_nt2048.yaml  \
                --use_ema --gradient_accumulation_steps 4   \
                    --output_dir output_mask     --tag scaleup_mp8_nt512_mask_data_10000_lr_1e-4
```
(2) evaluate the masks 
```
CUDA_VISIBLE_DEVICES=2 python TripoSG/scripts/eval/eval_triposg_mask.py \
    --output-path out_eval
    --router-ckpt  /data1/luoyajing/3d_pruning/output_mask/scaleup_mp8_nt512_mask_data_1000_lr_1e-3__mlp_sa_ca/checkpoints/000063.pt
    --thres "0.5"
```
