#!/bin/bash
set -x

EVAL_DATA_DIR=playground/data/eval_image/pope
OUTPUT_DIR=work_dirs/eval_output/pope
CKPT=${1}
# CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)
CKPT_NAME=$(echo $CKPT | awk -F'/' '{print $(NF-1) "-" $NF}')


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/${CKPT_NAME}/merge.jsonl

# judge if the number of json lines is 0
if [ ! -f "$output_file" ] || [ $(cat "$output_file" | wc -l) -eq 0 ]; then
    rm -f ${OUTPUT_DIR}/${CKPT_NAME}/*.jsonl
fi
# rm -f ${OUTPUT_DIR}/${SPLIT}/${CKPT_NAME}/*.jsonl

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m hicom.eval.image.inference_image_vqa \
            --benchmark pope \
            --model-path ${CKPT} \
            --image-folder ${EVAL_DATA_DIR}/val2014 \
            --question-file ${EVAL_DATA_DIR}/llava_pope_test.jsonl \
            --answer-file ${OUTPUT_DIR}/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --dtype bfloat16 &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
fi

python hicom/eval/image/eval_pope.py \
    --annotation-dir ${EVAL_DATA_DIR}/coco \
    --question-file ${EVAL_DATA_DIR}/llava_pope_test.jsonl \
    --result-file $output_file