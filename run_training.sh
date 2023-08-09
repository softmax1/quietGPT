python train.py \
config/train_shakespeare_char_softmax1.py \
--device=cuda \
--compile=False \
--use_softmax1=False \
--wandb_log=True \
--wandb_run_name='softmax1' \
--out_dir='baseline_out'