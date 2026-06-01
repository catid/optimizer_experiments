DEV_IDS=$1

DATA_DIR=/XXX/fineweb10B
WANDB_ENTITY=XXX
WANDB_PROJECT=XXX

N_WORKERS=$(echo -n "$DEV_IDS"|wc -m)
N_WORKERS=$(( ( N_WORKERS + 1 ) / 2))


###### EMA-Nesterov + Muon ######
MUON_LR=3.6e-4
ADAM_LR=3.6e-3
WARM=1800
REST=600
SEED=42
for EMA in 0 0.9 0.95 0.99 0.995 0.999; do
    for BETA in 0.1 0.3 0.5 0.7 0.9; do
        CUDA_VISIBLE_DEVICES=$DEV_IDS python -m torch.distributed.run --standalone --nproc_per_node $N_WORKERS train_gpt.py \
            --wandb_entity $WANDB_ENTITY \
            --wandb_project_name $WANDB_PROJECT \
            --input_bin "${DATA_DIR}/fineweb_train_*.bin" \
            --input_val_bin "${DATA_DIR}/fineweb_val_*.bin" \
            --model_name EMA-Nesterov+Muon \
            --optimizer muon \
            --lr $MUON_LR \
            --adam_lr $ADAM_LR \
            --momentum 0.95 \
            --adam_beta_1 0.9 \
            --adam_beta_2 0.95 \
            --use_ema_nesterov \
            --use_nesterov_step \
            --lookahead_stepsize $BETA \
            --lookahead_ema $EMA \
            --ema_nesterov_warmup $WARM \
            --ema_nesterov_rest $REST \
            --device_batch_size 32 \
            --batch_size 512 \
            --sequence_length 1024 \
            --num_iterations 6200 \
            --scheduler wsd_linear_decay \
            --warmup_iters 0 \
            --warmdown_iters 1800 \
            --adam_weight_decay 0 \
            --weight_decay 0 \
            --val_loss_every 200 \
            --use_nanogpt_weight_tying \
            --seed $SEED
    done
done

    # --eval_lerp_models \
    # --eval_lerp_every 200 \
    # --eval_lerp_gap 1 \


###### Pessimistic-Lookahead + Muon ######
MUON_LR=3.6e-4
ADAM_LR=3.6e-3
SEED=42
for K in 8 32 128 512; do
    for LSS in 0.5 0.7 0.9 1.1 1.3 1.5; do
        CUDA_VISIBLE_DEVICES=$DEV_IDS python -m torch.distributed.run --standalone --nproc_per_node $N_WORKERS train_gpt.py \
            --wandb_entity $WANDB_ENTITY \
            --wandb_project_name $WANDB_PROJECT \
            --input_bin "${DATA_DIR}/fineweb_train_*.bin" \
            --input_val_bin "${DATA_DIR}/fineweb_val_*.bin" \
            --model_name Pessimistic-Lookahead+Muon \
            --optimizer muon \
            --lr $MUON_LR \
            --adam_lr $ADAM_LR \
            --momentum 0.95 \
            --adam_beta_1 0.9 \
            --adam_beta_2 0.95 \
            --use_lookahead \
            --lookahead_step_size $LSS \
            --local_steps_K $K \
            --device_batch_size 32 \
            --batch_size 512 \
            --sequence_length 1024 \
            --num_iterations 6200 \
            --scheduler wsd_linear_decay \
            --warmup_iters 0 \
            --warmdown_iters 1800 \
            --adam_weight_decay 0 \
            --weight_decay 0 \
            --val_loss_every 200 \
            --use_nanogpt_weight_tying \
            --seed $SEED
    done
done

###### SNOO + Muon ######
MUON_LR=3.6e-4
ADAM_LR=3.6e-3
SEED=42
for SNOO_K in 10 50 100; do
    for SNOO_LR in 0.5 0.8 0.95; do
        for SNOO_MOM in 0.25 0.375 0.5 0.625 0.75; do
            CUDA_VISIBLE_DEVICES=$DEV_IDS python -m torch.distributed.run --standalone --nproc_per_node $N_WORKERS train_gpt.py \
                --wandb_entity $WANDB_ENTITY \
                --wandb_project_name $WANDB_PROJECT \
                --input_bin "${DATA_DIR}/fineweb_train_*.bin" \
                --input_val_bin "${DATA_DIR}/fineweb_val_*.bin" \
                --model_name SNOO+Muon \
                --optimizer muon \
                --lr $MUON_LR \
                --adam_lr $ADAM_LR \
                --momentum 0.95 \
                --adam_beta_1 0.9 \
                --adam_beta_2 0.95 \
                --use_snoo \
                --snoo_lr $SNOO_LR \
                --snoo_momentum $SNOO_MOM \
                --local_steps_K $SNOO_K \
                --device_batch_size 32 \
                --batch_size 512 \
                --sequence_length 1024 \
                --num_iterations 6200 \
                --scheduler wsd_linear_decay \
                --warmup_iters 0 \
                --warmdown_iters 1800 \
                --adam_weight_decay 0 \
                --weight_decay 0 \
                --val_loss_every 200 \
                --use_nanogpt_weight_tying \
                --seed $SEED
        done
    done
done


###### GPA + Muon ######
MUON_LR=3.6e-4
ADAM_LR=3.6e-3
SEED=42
for EMA_Y in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    for K in 8 16 32 64 128; do
        CUDA_VISIBLE_DEVICES=$DEV_IDS python -m torch.distributed.run --standalone --nproc_per_node $N_WORKERS train_gpt.py \
            --wandb_entity $WANDB_ENTITY \
            --wandb_project_name $WANDB_PROJECT \
            --input_bin "${DATA_DIR}/fineweb_train_*.bin" \
            --input_val_bin "${DATA_DIR}/fineweb_val_*.bin" \
            --model_name GPA+Muon \
            --optimizer muon \
            --lr $MUON_LR \
            --adam_lr $ADAM_LR \
            --momentum 0.95 \
            --adam_beta_1 0.9 \
            --adam_beta_2 0.95 \
            --use_gpa \
            --use_nesterov_step \
            --momentum_y $EMA_Y \
            --local_steps_K $K \
            --device_batch_size 32 \
            --batch_size 512 \
            --sequence_length 1024 \
            --num_iterations 6200 \
            --scheduler wsd_linear_decay \
            --warmup_iters 0 \
            --warmdown_iters 1800 \
            --adam_weight_decay 0 \
            --weight_decay 0 \
            --val_loss_every 200 \
            --use_nanogpt_weight_tying \
            --seed $SEED
    done
done