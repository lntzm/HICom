#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-4}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "which python: $(which python)"


# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=qwen2.5_7B-release
filename=$(basename -- "$0")
RUN_NAME="${filename%.*}"
OUTP_DIR=work_dirs


PRETRAIN_OUTPUT_DIR=${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME}/pretrain
mkdir -p $PRETRAIN_OUTPUT_DIR
if [ ! -f "${PRETRAIN_OUTPUT_DIR}/trainer_state.json" ]; then
    # Training Arguments
    GLOBAL_BATCH_SIZE=512
    LOCAL_BATCH_SIZE=32
    GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

    echo "Starting Pretrain"
    torchrun --nnodes $WORLD_SIZE \
        --nproc_per_node $NPROC_PER_NODE  \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank $RANK \
        hicom/train.py \
        --deepspeed scripts/zero0.json \
        --model_type hicom_qwen2 \
        --model_path playground/models/Qwen2.5-7B-Instruct \
        --vision_tower playground/models/siglip-so400m-patch14-384 \
        --mm_projector_type local43_global32 \
        --mm_tunable_parts mm_projector \
        --data_path scripts/data/pretrain.yaml \
        --data_folder playground/data \
        --mm_vision_select_layer -2 \
        --num_frames 1 \
        --max_num_frames 128 \
        --bf16 True \
        --tf32 True \
        --fp16 False \
        --output_dir ${PRETRAIN_OUTPUT_DIR} \
        --num_train_epochs 1 \
        --per_device_train_batch_size $LOCAL_BATCH_SIZE \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 5 \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --lazy_preprocess True \
        --report_to tensorboard \
        --run_name $RUN_NAME \
        2>&1 | tee -a "${PRETRAIN_OUTPUT_DIR}/training.log"
fi
if [ ! -f "${PRETRAIN_OUTPUT_DIR}/trainer_state.json" ]; then
    echo "Pretrain Failed"
    exit 1
fi


CONPRETRAIN_OUTPUT_DIR=${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME}/conditional_pretrain
mkdir -p $CONPRETRAIN_OUTPUT_DIR
if [ ! -f "${CONPRETRAIN_OUTPUT_DIR}/trainer_state.json" ]; then
    # Training Arguments
    GLOBAL_BATCH_SIZE=512
    LOCAL_BATCH_SIZE=8
    GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
    
    echo "Starting Conditional Pretrain"
    torchrun --nnodes $WORLD_SIZE \
        --nproc_per_node $NPROC_PER_NODE  \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank $RANK \
        hicom/train.py \
        --deepspeed scripts/zero0.json \
        --model_type hicom_qwen2 \
        --model_path playground/models/Qwen2.5-7B-Instruct \
        --vision_tower playground/models/siglip-so400m-patch14-384 \
        --use_guide direct \
        --mm_projector_type local43_global32_coarse \
        --pretrain_weights ${PRETRAIN_OUTPUT_DIR}/mm_projector.bin \
        --mm_tunable_parts mm_projector \
        --data_path scripts/data/conditional_pretrain.yaml \
        --data_folder playground/data \
        --mm_vision_select_layer -2 \
        --image_aspect_ratio pad \
        --mm_patch_merge_type spatial_unpad \
        --num_frames 32 \
        --max_num_frames 32 \
        --bf16 True \
        --tf32 True \
        --fp16 False \
        --output_dir ${CONPRETRAIN_OUTPUT_DIR} \
        --num_train_epochs 1 \
        --per_device_train_batch_size $LOCAL_BATCH_SIZE \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 5 \
        --learning_rate 1e-4 \
        --guide_injector_lr 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 12 \
        --lazy_preprocess True \
        --report_to tensorboard \
        --run_name $RUN_NAME \
        2>&1 | tee -a "${CONPRETRAIN_OUTPUT_DIR}/training.log"
fi
if [ ! -f "${CONPRETRAIN_OUTPUT_DIR}/trainer_state.json" ]; then
    echo "Conditional Pretrain Failed"
    exit 1
fi


IT_OUTPUT_DIR=${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME}/finetune
mkdir -p $IT_OUTPUT_DIR
if [ ! -f "${IT_OUTPUT_DIR}/trainer_state.json" ]; then
    # Training Arguments
    GLOBAL_BATCH_SIZE=256
    LOCAL_BATCH_SIZE=2
    GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

    echo "Starting SFT"
    torchrun --nnodes $WORLD_SIZE \
        --nproc_per_node $NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank $RANK \
        hicom/train.py \
        --deepspeed scripts/zero2.json \
        --model_type hicom_qwen2 \
        --model_path playground/models/Qwen2.5-7B-Instruct \
        --vision_tower playground/models/siglip-so400m-patch14-384 \
        --use_guide direct \
        --mm_projector_type local43_global32_coarse \
        --pretrain_weights ${CONPRETRAIN_OUTPUT_DIR}/mm_projector.bin \
        --mm_tunable_parts "mm_projector,language_model,vision_model_head,guide_encoder" \
        --data_path scripts/data/it_all_video.yaml \
        --data_folder playground/data \
        --mm_vision_select_layer -2 \
        --image_aspect_ratio pad \
        --mm_patch_merge_type spatial_unpad \
        --num_frames 32 \
        --max_num_frames 128 \
        --bf16 True \
        --tf32 True \
        --fp16 False \
        --output_dir $IT_OUTPUT_DIR \
        --num_train_epochs 1 \
        --per_device_train_batch_size $LOCAL_BATCH_SIZE \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 5 \
        --learning_rate 1e-5 \
        --vision_tower_lr 2e-6 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 12 \
        --report_to tensorboard \
        --run_name $RUN_NAME \
        2>&1 | tee -a "${IT_OUTPUT_DIR}/training.log"
fi
if [ ! -f "${IT_OUTPUT_DIR}/trainer_state.json" ]; then
    echo "SFT Failed"
    exit 1
fi