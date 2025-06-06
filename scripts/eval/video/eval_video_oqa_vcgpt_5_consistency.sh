#!/bin/bash
set -x

EVAL_DATA_DIR=playground/data/eval_video/Video-ChatGPT-eval
OUTPUT_DIR=work_dirs/eval_output/Video-ChatGPT-eval
CKPT=${1}
# CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)
CKPT_NAME=$(echo $CKPT | awk -F'/' '{print substr($NF, 1, 10) == "checkpoint" ? $(NF-3) "-" $(NF-2) "-" $(NF) : $(NF-2) "-" $(NF-1)}')


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/merge.json

# judge if the number of json lines is 0
if [ ! -f "$output_file" ] || [ $(cat "$output_file" | wc -l) -eq 0 ]; then
    rm -f ${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/*.json
fi

# if output_file not exists then inference
if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 hicom/eval/video/inference_video_oqa_vcgpt_consistency.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/Test_Videos \
            --question-file ${EVAL_DATA_DIR}/consistency_qa.json \
            --answer-file ${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --dtype bfloat16 &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi


AZURE_API_KEY=your_key
AZURE_API_ENDPOINT=your_endpoint
AZURE_API_DEPLOYNAME=your_deployname

python3 hicom/eval/video/eval_video_oqa_vcgpt_5_consistency.py \
    --pred-path ${output_file} \
    --output-dir ${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/gpt \
    --output-json ${OUTPUT_DIR}/answers/consistency/${CKPT_NAME}/results.json \
    --api-key $AZURE_API_KEY \
    --api-endpoint $AZURE_API_ENDPOINT \
    --api-deployname $AZURE_API_DEPLOYNAME \
    --num-tasks 4
