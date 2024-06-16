# export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="lora/uv"
export DATASET_DIR="/path/to/data/"

python3 train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --train_batch_size=4 \
  --rank=8 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10000 \
  --learning_rate=1e-4 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=1000 \
  --validation_prompt="A pig wearing blue overalls." \
  --seed=1337 \
  --resume_from_checkpoint latest 