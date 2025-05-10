#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1


PATH_TO_DATA=data_dir
CACHE_DIR=cache_dir

MODEL_TYPE=bert  
MODEL_SIZE=base 
MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
if [ $MODEL_TYPE = 'bert' ]
then
  MODEL_NAME=${MODEL_NAME}-uncased
fi



DATASET=SST-2  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI
THRESHOLDS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"


MODEL_PATH=model_dir

ALPHA=0.1
SIGNAL_TYPE='CAP'

PER_GPU_EVAL_BATCH_SIZE=64



for THRESHOLD in $THRESHOLDS;do
  echo $THRESHOLD
  python -um examples.eval_CAP \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $DATASET \
    --do_eval \
    --do_lower_case \
    --data_dir $PATH_TO_DATA/$DATASET \
    --output_dir $MODEL_PATH \
    --plot_data_dir ./plotting_${SIGNAL_TYPE}/ \
    --max_seq_length 128 \
    --early_exit_threshold $THRESHOLD \
    --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
    --cache_dir $CACHE_DIR \
    --alpha $ALPHA \
    --signal_type $SIGNAL_TYPE
done