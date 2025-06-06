#!/bin/bash
set -x

EVAL_DATA_DIR=playground/data/eval_video/EgoSchema
OUTPUT_DIR=work_dirs/eval_output/EgoSchema
CKPT=${1}
# CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)
CKPT_NAME=$(echo $CKPT | awk -F'/' '{print substr($NF, 1, 10) == "checkpoint" ? $(NF-3) "-" $(NF-2) "-" $(NF) : $(NF-2) "-" $(NF-1)}')


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/answers/${CKPT_NAME}/merge.csv

# judge if the number of json lines is 0
if [ ! -f "$output_file" ] || [ $(cat "$output_file" | wc -l) -eq 0 ]; then
    rm -f ${OUTPUT_DIR}/answers/${CKPT_NAME}/*.csv
fi

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 hicom/eval/video/inference_video_mcqa_egoschema.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/good_clips_git \
            --question-file ${EVAL_DATA_DIR}/questions.json \
            --answer-file ${OUTPUT_DIR}/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.csv \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --dtype bfloat16 &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    echo 'q_uid, answer' >> "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.csv >> "$output_file"
    done
fi

python3 hicom/eval/video/eval_video_mcqa_egoschema.py \
    --file $output_file