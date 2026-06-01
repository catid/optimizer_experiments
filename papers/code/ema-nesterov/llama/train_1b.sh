DEV_IDS=$1

ray stop
ray start --head

N_WORKERS=$(echo -n "$DEV_IDS"|wc -m)
N_WORKERS=$(( ( N_WORKERS + 1 ) / 2))

DATASET_PATH=/XXX/c4/en

WANDB_ENTITY=XXX
WANDB_PROJECT=XXX

##### Llama 1B #####
EMA=0.999
BETA=0.5
WARM=1500
REST=1000
CUDA_VISIBLE_DEVICES=$DEV_IDS python train_llama.py \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project_name $WANDB_PROJECT \
    --model_name EMA-Nesterov+Muon \
    --model_config $(pwd)/configs/llama_1b.json \
    --max_length 256 \
    --ray_use_gpu \
    --ray_num_workers $N_WORKERS \
    --workers 0 \
    --optimizer muon \
    --lr 0.05 \
    --adam_lr 0.05 \
    --momentum 0.95 \
    --adam_beta_1 0.9 \
    --adam_beta_2 0.95 \
    --use_ema_nesterov \
    --lookahead_stepsize $BETA \
    --lookahead_ema $EMA \
    --ema_nesterov_warmup $WARM \
    --ema_nesterov_rest $REST \
    --batch_size 128 \
    --total_batch_size 16384 \
    --num_training_steps 5000 \
    --save_logs_every 100 \
    --warmup_steps 500 \
    --decay_steps 500 \
    --eval_every 250 \
    --dtype fp32 \
    --amp \
    --scheduler wsd_quick_recovery \
    --eval_in_fp32 \
    --compile_model \
    --weight_decay 0 \
    --adam_weight_decay 0 \
    --dataset_path $DATASET_PATH
