#!/bin/bash

# === Configurable Parameters ===
JSON_FILE_PATH="./data/VidSum-Reason_mapping.json"  # Update as needed
VIDEO_DIR="./videos"                                # Update as needed
VIDEO_TYPE="mp4"
WORK_DIR="/path/to/work dictory"                    # Update as needed
IMAGE_PATH="/path/to/container_image.sqsh"          # Used only on SLURM
OPENAI_KEY=""

# === Detect if running under SLURM ===
IS_SLURM=${SLURM_JOB_ID:-""}

# Loop over all video-query pairs in the JSON
for vidQry in $(jq -r 'keys_unsorted | .[]' "$JSON_FILE_PATH"); do
    video_id=$(jq -r ".\"$vidQry\".video_id" "$JSON_FILE_PATH")
    query=$(jq -r ".\"$vidQry\".query" "$JSON_FILE_PATH")

    echo "Running prediction for VidQry: $vidQry [Video: $video_id | Query: $query]"

    PYTHON_CMD="python src/model/solver.py \
        --video_name \"$video_id\" \
        --video_type \"$VIDEO_TYPE\" \
        --video_dir \"$VIDEO_DIR\" \
        --work_dir \"$WORK_DIR\" \
        --openai_key \"$OPENAI_KEY\""
        
    if [ -n "$IS_SLURM" ]; then
        # SLURM: run inside container
        srun --gpus=1 \
             --container-image=$IMAGE_PATH \
             --container-mounts=$(pwd):/workspace \
             /bin/bash -c "cd /workspace && $PYTHON_CMD"
    else
        # Local: run directly
        eval $PYTHON_CMD
    fi
donec