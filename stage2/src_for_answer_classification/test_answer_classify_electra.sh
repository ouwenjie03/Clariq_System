#!/usr/bin/env bash

DATATYPE=answer_data
MODELTYPE=electra_large

python run_bertology_classify.py \
--do predict_pair \
--output_dir ./models/${DATATYPE}_${MODELTYPE} \
--test_data_file ../data/answer_datas/dev.pkl \
--data_type ${DATATYPE} \
--model_type ${MODELTYPE} \
--model_name_or_path ./models/answer_data_electra_large/ \
--tokenizer_name_or_path /storage08/PLM/english_electra/electra-large-discriminator-google/ \
--per_device_eval_batch_size 64 \
--max_len 256 \
--num_labels 3 \
--seed 12345
