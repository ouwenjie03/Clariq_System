#!/usr/bin/env bash

DATATYPE=answer_data
MODELTYPE=roberta_large

python run_bertology_classify.py \
--do predict_pair \
--output_dir ./models/${DATATYPE}_${MODELTYPE} \
--test_data_file ../data/answer_datas/dev.pkl \
--data_type ${DATATYPE} \
--model_type ${MODELTYPE} \
--model_name_or_path ./models/answer_data_roberta_large/ \
--tokenizer_name_or_path  /storage08/PLM/english_roberta/roberta_large_huggingface/ \
--per_device_eval_batch_size 64 \
--max_len 256 \
--num_labels 3 \
--seed 12345
