python train.py \
config/train_shakespeare_char_softmax1.py \
--device=cuda \
--use_softmax1=True \
--wandb_log=True \
--wandb_run_name='quiet_gpt_softmax1_without_zero_shift_non_scaled_dot_product_attention'