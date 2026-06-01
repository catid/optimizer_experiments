DEV_IDS=$1

DATA_DIR=/XXX/fineweb10B
WANDB_ENTITY=XXX
WANDB_PROJECT=XXX

N_WORKERS=$(echo -n "$DEV_IDS"|wc -m)
N_WORKERS=$(( ( N_WORKERS + 1 ) / 2))


###### EMA-Nesterov + Muon ######
MUON_LR=3.6e-4
ADAM_LR=3.6e-3
BETA=0.5
EMA=0.995
WARM=1800
REST=600
for SEED in {0..4}; do
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
    # --eval_lerp_models \
    # --eval_lerp_every 200 \
    # --eval_lerp_gap 1 \


###### EMA-Nesterov + NorMuon ######
MUON_LR=3.6e-4
ADAM_LR=3.6e-3
BETA=0.5
EMA=0.995
WARM=1800
REST=600
SEED=42
for SEED in {0..4}; do
    CUDA_VISIBLE_DEVICES=$DEV_IDS python -m torch.distributed.run --standalone --nproc_per_node $N_WORKERS train_gpt.py \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project_name $WANDB_PROJECT \
        --input_bin "${DATA_DIR}/fineweb_train_*.bin" \
        --input_val_bin "${DATA_DIR}/fineweb_val_*.bin" \
        --model_name EMA-Nesterov+NorMuon \
        --optimizer normuon \
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

###### EMA-Nesterov + SOAP ######
LR=1e-3
BETA=0.5
EMA=0.995
WARM=1800
REST=600
SEED=42
for SEED in {0..4}; do
    CUDA_VISIBLE_DEVICES=$DEV_IDS python -m torch.distributed.run --standalone --nproc_per_node $N_WORKERS train_gpt.py \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project_name $WANDB_PROJECT \
        --input_bin "${DATA_DIR}/fineweb_train_*.bin" \
        --input_val_bin "${DATA_DIR}/fineweb_val_*.bin" \
        --model_name EMA-Nesterov+SOAP \
        --optimizer soap \
        --lr $LR \
        --adam_beta_1 0.95 \
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


###### EMA-Nesterov + Adam ######
LR=1e-3
BETA=0.7
EMA=0.99
WARM=1800
REST=600
SEED=42
for SEED in {0..4}; do
    CUDA_VISIBLE_DEVICES=$DEV_IDS python -m torch.distributed.run --standalone --nproc_per_node $N_WORKERS train_gpt.py \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project_name $WANDB_PROJECT \
        --input_bin "${DATA_DIR}/fineweb_train_*.bin" \
        --input_val_bin "${DATA_DIR}/fineweb_val_*.bin" \
        --model_name EMA-Nesterov+Adam \
        --optimizer adam \
        --lr $LR \
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
