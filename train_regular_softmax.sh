python train.py \
config/train_shakespeare_char_regular_softmax.py \
--device=cuda \
--compile=False \
--eval_iters=20 \
--log_interval=1 \
--block_size=64 \
--batch_size=12 \
--n_layer=4 \
--n_head=4 \
--n_embd=128 \
--max_iters=2000 \
--lr_decay_iters=2000 \
--dropout=0.0 \
--wandb_log=False \
--use_softmax1=False