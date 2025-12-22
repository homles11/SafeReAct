export WANDB_MODE=offline
export MASTER_PORT=$((29002 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

### Llama-3-8B Config ###
model_name_or_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
refmodel_path=./model/prune_reasoning
lorra_alpha=10
layers="5,10,15,20,25,27"
transform_layers="-1"

output_dir="./SafeReAct_R1_out/2Model_Qwen7B_CB_R1safeRR_2alpha_$lorra_alpha"

echo "model_name_or_path=$model_name_or_path"
echo "refmodel_path=$refmodel_path"
echo "output_dir=$output_dir"

accelerate launch --config_file configs/accelerate_zero1.yaml \
    --num_processes 1 --main_process_port $MASTER_PORT --deepspeed_hostfile ds_hostfile \
    src/lorra_safereact_twomodel.py \
    --model_name_or_path $model_name_or_path \
    --refmodel_path $refmodel_path \
    --target_layers $layers \
    --transform_layers $transform_layers \
    --lorra_alpha $lorra_alpha \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir  $output_dir \
    --overwrite_output_dir \
    --max_steps 400 \
    --bf16 True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --eval_strategy 'no' \
    --eval_steps 1000  \
    --save_steps 50 \
    --save_total_limit 6 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 8192 \
    --q_lora False \
    --gradient_checkpointing True \
    --report_to none \
    --log_every 1