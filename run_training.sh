python train.py \
config/train_shakespeare_char_softmax1.py \
--device=mps \
--compile=False \
--use_softmax1=True \
--wandb_log=False \
--wandb_run_name='quiet_gpt_softmax1_without_zero_shift_non_scaled_dot_product_attention'