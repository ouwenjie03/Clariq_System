#!/usr/bin/env bash

DATATYPE=answer_data
MODELTYPE=electra_large

python run_bertology_classify.py \
--train_data_file ../data/answer_datas/train.pkl \
--eval_data_file ../data/answer_datas/dev.pkl \
--output_dir ./models/${DATATYPE}_${MODELTYPE} \
--logging_dir ./logs/${DATATYPE}_${MODELTYPE} \
--data_type ${DATATYPE} \
--model_type ${MODELTYPE} \
--model_name_or_path /storage08/PLM/english_electra/electra-large-discriminator-google/ \
--do_train \
--do_eval \
--learning_rate 5e-6 \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 32 \
--max_len 256 \
--num_labels 3 \
--num_train_epochs 10 \
--max_grad_norm 5 \
--logging_steps 300 \
--save_steps 300 \
--eval_steps 300 \
--evaluate_during_training \
--overwrite_output_dir \
--logging_first_step \
--seed 12345
