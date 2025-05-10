#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

PATH_TO_DATA=data_dir
CACHE_DIR=cache_dir

MODEL_TYPE=bert  
MODEL_SIZE=base  
MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}-uncased



DATASET=SST-2  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

EPOCHS=10
PER_GPU_TRAIN_BATCH_SIZE=64
LR=2e-5

PER_GPU_EVAL_BATCH_SIZE=32
SAVE_STEPS=500



      
      
python -um examples.train \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME \
  --task_name $DATASET \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --output_dir ./saved_models/$DATASET \
  --max_seq_length 128 \
  --cache_dir $CACHE_DIR \
  --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
  --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $EPOCHS \
  --overwrite_output_dir \
  --seed 42 \
  --save_steps $SAVE_STEPS