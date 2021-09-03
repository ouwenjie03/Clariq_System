#!/usr/bin/env python
# encoding: utf-8

"""
@author: ouwenjie
@license: NeteaseGame Licence 
@contact: ouwenjie@corp.netease.com


"""

import dataclasses
import logging
import os
import _pickle as pickle
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

import torch

from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from transformers.trainer import SequentialDistributedSampler
from transformers import DataCollator
from transformers import EvalPrediction



logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: gpt2"},
    )
    config_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    add_train_data_file1: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    add_train_data_file2: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to output result on (a text file)."},
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    data_type: Optional[str] = field(
        default=None,
        metadata={"help": "pair or sent"}
    )
    max_len: int = field(
        default=256,
        metadata={
            "help": "sentence max length"
        }
    )
    result_out_file: Optional[str] = field(
        default=None, metadata={"help": "The output data file (a text file)."}
    )
    do: Optional[str] = field(
        default='train', metadata={"help": "The output data file (a text file)."}
    )
    num_labels: int = field(
        default=2,
        metadata={
            "help": "label num"
        }
    )


def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        label_set = list(set(labels) - set([0]))
        n_labels = len(set(labels))
        if n_labels == 2:
            f1 = f1_score(y_true=labels, y_pred=preds)
            r = recall_score(y_true=labels, y_pred=preds)
            p = precision_score(y_true=labels, y_pred=preds)
            auc = roc_auc_score(y_true=labels, y_score=preds)
            return {'p': p, 'r': r, 'f1': f1, 'auc': auc}
        else:
            p = precision_score(y_true=labels, y_pred=preds, average='weighted', labels=label_set)
            r = recall_score(y_true=labels, y_pred=preds, average='weighted', labels=label_set)
            f1 = f1_score(y_true=labels, y_pred=preds, average='weighted', labels=label_set)
            return {'p': p, 'r': r, 'f1': f1}

    return compute_metrics_fn


@dataclass
class MyDataCollator:
    def __call__(self, datas) -> Dict[str, torch.Tensor]:
        dict_datas = {
            'input_ids': torch.tensor([_d['input_ids'] for _d in datas], dtype=torch.long),
            'attention_mask': torch.tensor([_d['attention_mask'] for _d in datas], dtype=torch.bool),
            'labels': torch.tensor([_d['label'] for _d in datas], dtype=torch.long),
        }
        if 'token_type_ids' in datas[0] and datas[0]['token_type_ids'] is not None:
            dict_datas['token_type_ids'] = torch.tensor([_d['token_type_ids'] for _d in datas], dtype=torch.long)
        return dict_datas


class MyDataset(Dataset):
    def __init__(self, tokenizer, max_len):
        self.datas = []
        self.tokenizer: RobertaTokenizer = tokenizer
        self.max_len = max_len

    def format_one_data(self, p1, p2):
        tokens1 = self.tokenizer.tokenize(p1)
        input_tokens = ['[CLS]'] + tokens1 + ['[SEP]']
        if p2 is not None:
            tokens2 = self.tokenizer.tokenize(p2)
            input_tokens += tokens2 + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1 for _ in range(len(input_ids))]
        token_type_ids = [0 for _ in range(len(tokens1) + 2)]
        if p2 is not None:
            token_type_ids += [1 for _ in range(len(tokens2) + 1)]

        # pad
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [0 for _ in range(pad_len)]
        attention_mask = attention_mask + [0 for _ in range(pad_len)]
        token_type_ids = token_type_ids + [0 for _ in range(pad_len)]

        data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }
        return data

    def format_one_data_roberta(self, p1, p2):
        input_ids = self.tokenizer.encode(p1, p2, max_length=self.max_len, pad_to_max_length=True)
        attention_mask = [0 if _id == self.tokenizer.pad_token_id else 1 for _id in input_ids]

        data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        return data

    def load_pair_datas(self, df, model_type):
        for p1, p2, label in tqdm(df[['initial_request', 'question', 'label']].values):
            p1 = p1.lower()
            p2 = p2.lower()
            if 'roberta' in model_type:
                data = self.format_one_data_roberta(p1, p2)
            else:
                data = self.format_one_data(p1, p2)
            data['label'] = label
            self.datas.append(data)

    def load_sent_datas(self, df):
        for p1, label in tqdm(df[['initial_request', 'label']].values):
            p1 = p1.lower()
            data = self.format_one_data(p1, None)
            data['label'] = label
            self.datas.append(data)

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return len(self.datas)


def get_dataset(args: DataTrainingArguments, tokenizer, df, model_type):
    dataset = MyDataset(tokenizer, max_len=args.max_len)
    if 'pair' in args.data_type:
        dataset.load_pair_datas(df, model_type)
    else:
        dataset.load_sent_datas(df)
    print(tokenizer.convert_ids_to_tokens([_c for _c in dataset[0]['input_ids'] if _c != tokenizer.pad_token_id]))
    print(dataset[0]['label'])

    return dataset


def main():
    # See all possible arguments in transformers.training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.do == 'predict_pair':
        predict_pair(model_args, data_args, training_args)
    else:
        train(model_args, data_args, training_args)


def train(model_args, data_args, training_args):
    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logs
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if 'roberta' in model_args.model_type:
        tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path)
        config = RobertaConfig.from_pretrained(model_args.model_name_or_path)
        config.num_labels = data_args.num_labels
        model = RobertaForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    elif 'electra' in model_args.model_type:
        tokenizer = ElectraTokenizer.from_pretrained(model_args.model_name_or_path)
        config = ElectraConfig.from_pretrained(model_args.model_name_or_path)
        config.num_labels = data_args.num_labels
        model = ElectraForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        # default -> bert
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
        config = BertConfig.from_pretrained(model_args.model_name_or_path)
        config.num_labels = data_args.num_labels
        model = BertForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
        tokenizer.add_special_tokens()

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    train_df = pd.read_csv(data_args.train_data_file, sep='\t')
    if data_args.add_train_data_file1 is not None:
        tmp = pd.read_csv(data_args.add_train_data_file1, sep='\t')
        train_df = pd.concat([train_df, tmp])
    if data_args.add_train_data_file2 is not None:
        tmp = pd.read_csv(data_args.add_train_data_file2, sep='\t')
        train_df = pd.concat([train_df, tmp])
    train_df = train_df.fillna('no_q')
    train_dataset = get_dataset(data_args, tokenizer, train_df, model_args.model_type) if training_args.do_train else None

    dev_df = pd.read_csv(data_args.eval_data_file, sep='\t')
    dev_df = dev_df.fillna('no_q')
    eval_dataset = get_dataset(data_args, tokenizer, dev_df, model_args.model_type) if training_args.do_eval else None
    data_collator = MyDataCollator()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(),
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        trainer.compute_metrics = build_compute_metrics_fn()
        result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def predict_pair(model_args, data_args, training_args):
    # Set seed
    set_seed(training_args.seed)

    if 'roberta' in model_args.model_type:
        tokenizer = RobertaTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
        config = RobertaConfig.from_pretrained(model_args.model_name_or_path)
        config.num_labels = data_args.num_labels
        model = RobertaForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    elif 'electra' in model_args.model_type:
        tokenizer = ElectraTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
        config = ElectraConfig.from_pretrained(model_args.model_name_or_path)
        config.num_labels = data_args.num_labels
        model = ElectraForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        # default -> bert
        tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
        config = BertConfig.from_pretrained(model_args.model_name_or_path)
        config.num_labels = data_args.num_labels
        model = BertForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)

    model.to(training_args.device)

    test_df = pd.read_csv(data_args.test_data_file, sep='\t')
    test_df.fillna('no_q', inplace=True)
    test_dataset = get_dataset(data_args, tokenizer, test_df, model_args.model_type)
    data_collator = MyDataCollator()
    if training_args.local_rank != -1:
        sampler = SequentialDistributedSampler(test_dataset)
        model = torch.nn.DataParallel(model)
    else:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        sampler = SequentialSampler(test_dataset)
    print(len(test_dataset))
    dataloader = DataLoader(
        test_dataset,
        sampler=sampler,
        batch_size=training_args.eval_batch_size,
        collate_fn=data_collator,
    )

    model.eval()
    all_probs = []
    for inputs in tqdm(dataloader):
        for k, v in inputs.items():
            inputs[k] = v.to(training_args.device)
        inputs.pop('labels')
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs[0]
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())

    testset = pd.read_csv(data_args.test_data_file, sep='\t')
    testset['probs'] = all_probs

    testset[['tid', 'qid', 'probs']].to_csv('./{}_{}_result.tsv'.format(data_args.data_type, model_args.model_type), sep='\t', index=None)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
