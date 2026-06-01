DEV_IDS=$1

ray stop
ray start --head

N_WORKERS=$(echo -n "$DEV_IDS"|wc -m)
N_WORKERS=$(( ( N_WORKERS + 1 ) / 2))

DATASET_PATH=/XXX/c4/en

WANDB_ENTITY=XXX
WANDB_PROJECT=XXX

##### Llama 60M #####
EMA=0.99
BETA=0.7
WARM=750
REST=500
TOTAL_BATCHES=(2048 4096 8192 16384)
LRS=(0.03 0.0357 0.0424 0.0505) # **(1/4) rule
for i in "${!TOTAL_BATCHES[@]}"; do
    CUDA_VISIBLE_DEVICES=$DEV_IDS python train_llama.py \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project_name $WANDB_PROJECT \
        --model_name EMA-Nesterov+Muon \
        --model_config $(pwd)/configs/llama_60m.json \
        --max_length 256 \
        --ray_use_gpu \
        --ray_num_workers $N_WORKERS \
        --workers 0 \
        --optimizer muon \
        --lr ${LRS[i]} \
        --adam_lr ${LRS[i]} \
        --momentum 0.95 \
        --adam_beta_1 0.9 \
        --adam_beta_2 0.95 \
        --use_ema_nesterov \
        --lookahead_stepsize $BETA \
        --lookahead_ema $EMA \
        --ema_nesterov_warmup $WARM \
        --ema_nesterov_rest $REST \
        --batch_size 128 \
        --total_batch_size ${TOTAL_BATCHES[i]} \
        --num_training_steps 2500 \
        --save_logs_every 100 \
        --warmup_steps 250 \
        --decay_steps 250 \
        --eval_every 100 \
        --dtype fp32 \
        --amp \
        --scheduler wsd_quick_recovery \
        --eval_in_fp32 \
        --compile_model \
        --weight_decay 0 \
        --adam_weight_decay 0 \
        --dataset_path $DATASET_PATH
done

##### Llama 130M #####
EMA=0.99
BETA=0.7
WARM=750
REST=500
TOTAL_BATCHES=(4096 8192 16384 32768)
LRS=(0.0424 0.0505 0.06 0.0714) # **(1/4) rule
for i in "${!TOTAL_BATCHES[@]}"; do
    CUDA_VISIBLE_DEVICES=$DEV_IDS python train_llama.py \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project_name $WANDB_PROJECT \
        --model_name EMA-Nesterov+Muon \
        --model_config $(pwd)/configs/llama_130m.json \
        --max_length 256 \
        --ray_use_gpu \
        --ray_num_workers $N_WORKERS \
        --workers 0 \
        --optimizer muon \
        --lr ${LRS[i]} \
        --adam_lr ${LRS[i]} \
        --momentum 0.95 \
        --adam_beta_1 0.9 \
        --adam_beta_2 0.95 \
        --use_ema_nesterov \
        --lookahead_stepsize $BETA \
        --lookahead_ema $EMA \
        --ema_nesterov_warmup $WARM \
        --ema_nesterov_rest $REST \
        --batch_size 128 \
        --total_batch_size ${TOTAL_BATCHES[i]} \
        --num_training_steps 2500 \
        --save_logs_every 100 \
        --warmup_steps 250 \
        --decay_steps 250 \
        --eval_every 100 \
        --dtype fp32 \
        --amp \
        --scheduler wsd_quick_recovery \
        --eval_in_fp32 \
        --compile_model \
        --weight_decay 0 \
        --adam_weight_decay 0 \
        --dataset_path $DATASET_PATH
done

##### Llama 350M #####
EMA=0.99
BETA=0.5
WARM=750
REST=500
TOTAL_BATCHES=(12288 24576 49152 98304)
LRS=(0.03 0.0357 0.0424 0.0505) # **(1/4) rule
for i in "${!TOTAL_BATCHES[@]}"; do
    CUDA_VISIBLE_DEVICES=$DEV_IDS python train_llama.py \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project_name $WANDB_PROJECT \
        --model_name EMA-Nesterov+Muon \
        --model_config $(pwd)/configs/llama_350m.json \
        --max_length 256 \
        --ray_use_gpu \
        --ray_num_workers $N_WORKERS \
        --workers 0 \
        --optimizer muon \
        --lr ${LRS[i]} \
        --adam_lr ${LRS[i]} \
        --momentum 0.95 \
        --adam_beta_1 0.9 \
        --adam_beta_2 0.95 \
        --use_ema_nesterov \
        --lookahead_stepsize $BETA \
        --lookahead_ema $EMA \
        --ema_nesterov_warmup $WARM \
        --ema_nesterov_rest $REST \
        --batch_size 128 \
        --total_batch_size ${TOTAL_BATCHES[i]} \
        --num_training_steps 2500 \
        --save_logs_every 100 \
        --warmup_steps 250 \
        --decay_steps 250 \
        --eval_every 100 \
        --dtype fp32 \
        --amp \
        --scheduler wsd_quick_recovery \
        --eval_in_fp32 \
        --compile_model \
        --weight_decay 0 \
        --adam_weight_decay 0 \
        --dataset_path $DATASET_PATH
done