python train.py \
config/train_shakespeare_char_softmax1.py \
--device=mps \
--compile=False \
--use_softmax1=True \
--wandb_log=True \
--wandb_run_name='softmax1_trial_measure_act_kurtosis'