cd src/open-r1-multimodal
export PYTHONPATH=./src

export DEBUG_MODE="true"

TIMESTAMP=$(date +%y%m%d-%H%M)
RUN_NAME="Qwen2.5-VL-7B-GRPO-UI-TRANSITION-$TIMESTAMP-data-2000-k-1-NO-REASONING"
export LOG_PATH="logs/debug_log_$RUN_NAME.txt"
exec > >(tee -a "$LOG_PATH") 2>&1

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_uishift.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /model/Qwen2.5-VL-7B-Instruct \
    --dataset_name none \
    --image_folders /data \
    --data_file_paths data_jsonl/ui_transition_training_2000_k_1_no_reasoning.jsonl \
    --freeze_vision_modules true \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 8 \
    --run_name $RUN_NAME \
    --save_steps 240 \
    --save_only_model true