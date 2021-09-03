#!/usr/bin/env bash

DATATYPE=${1}
MODELTYPE=electra_large_add_pseudo

python run_bertology_classify.py \
--do predict_pair \
--output_dir ./models/${DATATYPE}_${MODELTYPE} \
--test_data_file ../data/pair_datas_src/${DATATYPE}.tsv \
--data_type ${DATATYPE} \
--model_type ${MODELTYPE} \
--model_name_or_path ../models/pair_data_src_add_pseudo_electra_large/ \
--tokenizer_name_or_path /storage08/PLM/english_electra/electra-large-discriminator-google/ \
--per_device_eval_batch_size 64 \
--max_len 256 \
--num_labels 2 \
--seed 12345
