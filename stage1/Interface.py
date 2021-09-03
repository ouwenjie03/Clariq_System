#!/usr/bin/env python
# encoding: utf-8

"""
@author: ouwenjie
@license: NeteaseGame Licence 
@contact: ouwenjie@corp.netease.com


"""

import os
import re
import time
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem.porter import PorterStemmer

import _pickle as pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from transformers import ElectraForSequenceClassification, ElectraConfig, ElectraTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer
from BertForMultitask import ElectraForSequenceClassificationMultiTask


nltk.data.path.append('./data/nltk_data')

def stem_tokenize(text, remove_stopwords=True):
    stemmer = PorterStemmer()
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    return [stemmer.stem(word) for word in tokens]


class Config:
    def __init__(self):
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        self.nltk_data_dir = os.path.join(self.abs_path, 'data/nltk_data')
        self.question_bank_file_path = os.path.join(self.abs_path, 'data/question_bank.tsv')
        self.train_single_turn_file_path = os.path.join(self.abs_path, 'data/train.tsv')

        self.model_path_for_clariq = os.path.join(self.abs_path, 'models/electra_for_clarify_question_select')
        self.model_path_for_clariq_multitask = os.path.join(self.abs_path, 'models/electra_multitask_for_clarify_question_select')
        self.model_path_for_answer = os.path.join(self.abs_path, 'models/electra_for_answer_classification')

        self.num_labels_for_clariq = 2
        self.num_labels_for_answer = 3
        self.num_regress = 3

        self.clariq_model_type = 'electra'
        self.clariq_multitask_model_type = 'electra_multitask'
        self.answer_model_type = 'electra'


class Interface:
    def __init__(self, interface_config):
        self.interface_config = interface_config
        self.max_len = 256
        self.batch_size = 1024
        self.device = 'cuda'
        self.question_bank = pd.read_csv(interface_config.question_bank_file_path, sep='\t')
        self.question_bank.fillna('no_q', inplace=True)
        
        # bm25
        print('loading bm25 models...')
        self.bm25, self.bm25_corpus = self._init_bm25_model()
        self.train_bm25, self.train_bm25_corpus = self._init_traindf_bm25_model()
        self.notin_train_questions = self._init_notin_train_questions()

        # electra
        print('loading clariq models...')
        self.config_for_clariq, \
        self.tokenizer_for_clariq, \
        self.model_for_clariq = self._init_deep_model(
            interface_config.clariq_model_type,
            interface_config.model_path_for_clariq,
            interface_config.num_labels_for_clariq
        )

        print('loading clariq classify multitask models...')
        self.multitask_config_for_answer, \
        self.multitask_tokenizer_for_answer, \
        self.multitask_model_for_clariq = self._init_deep_model(
            interface_config.clariq_multitask_model_type,
            interface_config.model_path_for_clariq_multitask,
            interface_config.num_labels_for_clariq,
            interface_config.num_regress
        )

        print('loading answer classify models...')
        self.config_for_answer, \
        self.tokenizer_for_answer, \
        self.model_for_answer = self._init_deep_model(
            interface_config.answer_model_type,
            interface_config.model_path_for_answer,
            interface_config.num_labels_for_answer
        )

    def _init_deep_model(self, model_type, model_path, num_labels, num_regs=None):
        if 'roberta' in model_type:
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
            config = RobertaConfig.from_pretrained(model_path)
            config.num_labels = num_labels
            model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)
            model.eval()
            model.to(self.device)
        elif 'electra_multitask' in model_type:
            tokenizer = ElectraTokenizer.from_pretrained(model_path)
            tokenizer.add_special_tokens({'additional_special_tokens': ['[VALUES]']})
            config = ElectraConfig.from_pretrained(model_path)
            config.num_labels = num_labels
            config.num_regs = num_regs
            config.vocab_size = len(tokenizer)
            model = ElectraForSequenceClassificationMultiTask.from_pretrained(model_path, config=config)
            model.eval()
            model.to(self.device)
        elif 'electra' in model_type:
            tokenizer = ElectraTokenizer.from_pretrained(model_path)
            config = ElectraConfig.from_pretrained(model_path)
            config.num_labels = num_labels
            model = ElectraForSequenceClassification.from_pretrained(model_path, config=config)
            model.eval()
            model.to(self.device)
        else:
            raise NotImplementedError()
        return config, tokenizer, model

    def _init_traindf_bm25_model(self):
        train_df = pd.read_csv(self.interface_config.train_single_turn_file_path, sep='\t')
        train_df = train_df.drop_duplicates(['topic_id', 'question_id']).reset_index(drop=True).fillna('no_q')

        added_tokens = []
        added_cnames = ['initial_request', 'answer', 'topic_desc']
        for qid in self.question_bank['question_id'].values:
            words = []
            for cname in added_cnames:
                irs = train_df[train_df['question_id'] == qid][cname].unique()
                # irs = all_df[all_df['question_id'] == qid][cname].unique()
                for ir in irs:
                    ws = stem_tokenize(ir)
                    words.extend(ws)
            words = list(set(words))
            added_tokens.append(words)
        self.question_bank['tokens_from_train'] = added_tokens
        self.question_bank['all_tokens'] = self.question_bank['tokenized_question_list'] + self.question_bank['tokens_from_train']
        self.question_bank['all_token_str'] = self.question_bank['all_tokens'].map(lambda x: ' '.join(x))

        # add train_df initial_request tokens
        # bm25_corpus = question_bank['tokenized_question_list'].tolist()
        bm25_corpus = self.question_bank['all_tokens'].tolist()
        bm25 = BM25Okapi(bm25_corpus)

        return bm25, bm25_corpus

    def _init_bm25_model(self):
        self.question_bank['tokenized_question_list'] = self.question_bank['question'].map(stem_tokenize)
        self.question_bank['tokenized_question_str'] = self.question_bank['tokenized_question_list'].map(lambda x: ' '.join(x))

        bm25_corpus = self.question_bank['tokenized_question_list'].tolist()
        bm25 = BM25Okapi(bm25_corpus)
        return bm25, bm25_corpus

    def _init_notin_train_questions(self):
        notin_train = self.question_bank['question_id'].values.tolist()
        notin_train = sorted(notin_train, key=lambda x: len(self.question_bank[self.question_bank['question_id'] == x]['tokenized_question_str'].values[0]))
        notin_train = [self.question_bank[self.question_bank['question_id'] == x]['question'].values[0] for x in notin_train]
        return notin_train

    def _format_bert_datas(self, tokenizer, datas, model_type='electra'):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for d in datas:
            if len(d) == 1:
                tok = tokenizer.tokenize(d[0])[:self.max_len-2]
                tok = ['[CLS]'] + tok + ['[SEP]']
                inpids = tokenizer.convert_tokens_to_ids(tok)
                attm = [1 for _ in range(len(tok))]
                ttids = [0 for _ in range(len(tok))]
            else:
                p1 = d[0]
                p2 = d[1]
                tok1 = tokenizer.tokenize(p1)
                tok2 = tokenizer.tokenize(p2)
                if 'multitask' in model_type:
                    tok = ['[CLS]', '[VALUES]'] + tok1 + ['[SEP]'] + tok2 + ['[SEP]']
                    while len(tok1) + len(tok2) > self.max_len - 4:
                        if len(tok1) > len(tok2):
                            tok1 = tok1[:-1]
                        else:
                            tok2 = tok2[:-1]
                    inpids = tokenizer.convert_tokens_to_ids(tok)
                    attm = [1 for _ in range(len(tok))]
                    ttids = [0 for _ in range(len(tok1) + 3)] + [1 for _ in range(len(tok2) + 1)]
                else:
                    tok = ['[CLS]'] + tok1 + ['[SEP]'] + tok2 + ['[SEP]']
                    while len(tok1) + len(tok2) > self.max_len - 3:
                        if len(tok1) > len(tok2):
                            tok1 = tok1[:-1]
                        else:
                            tok2 = tok2[:-1]

                    inpids = tokenizer.convert_tokens_to_ids(tok)
                    attm = [1 for _ in range(len(tok))]
                    ttids = [0 for _ in range(len(tok1) + 2)] + [1 for _ in range(len(tok2) + 1)]

            # pad
            pad_len = self.max_len - len(inpids)
            inpids += [0 for _ in range(pad_len)]
            attm += [0 for _ in range(pad_len)]
            ttids += [0 for _ in range(pad_len)]

            input_ids.append(inpids)
            attention_mask.append(attm)
            token_type_ids.append(ttids)

        return input_ids, attention_mask, token_type_ids

    def _format_roberta_datas(self, tokenizer, datas):
        input_ids = []
        attention_mask = []
        for d in datas:
            p1 = d[0]
            p2 = None if len(d) == 1 else d[1]
            inpids = tokenizer.encode(p1, p2, max_length=self.max_len, pad_to_max_length=True, truncation=True)
            attm = [0 if _id == tokenizer.pad_token_id else 1 for _id in inpids]

            input_ids.append(inpids)
            attention_mask.append(attm)

        return input_ids, attention_mask, None

    def _get_model_output_probs(self, model, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.bool, device=self.device)
            token_type_ids_tensor = None if token_type_ids is None else torch.tensor(token_type_ids, dtype=torch.long, device=self.device)
            outputs = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                token_type_ids=token_type_ids_tensor
            )
            logits = outputs[0]
            probs = torch.softmax(logits, dim=-1)[:, 1]
            # probs = probs.cpu().numpy().tolist()

        return probs

    def _get_bm25_candidate_answer(self, input_data):
        query_str = input_data['initial_request']
        for ctx in input_data['conversation_context']:
            query_str += ctx['answer']

        bm25_ranked_list = self.bm25.get_top_n(stem_tokenize(query_str, True), self.bm25_corpus, n=200)
        bm25_q_list = [' '.join(sent) for sent in bm25_ranked_list]
        bm25_preds = self.question_bank.set_index('tokenized_question_str').loc[bm25_q_list, 'question'].tolist()
        return bm25_preds

    def _get_new_bm25_candidate_answer(self, query_str, topn=200):
        bm25_ranked_list = self.train_bm25.get_top_n(stem_tokenize(query_str, True), self.train_bm25_corpus, n=topn//2)
        bm25_q_list = [' '.join(sent) for sent in bm25_ranked_list]
        bm25_preds = self.question_bank.set_index('all_token_str').loc[bm25_q_list, 'question'].tolist()

        # bm25_ranked_list = self.bm25.get_top_n(stem_tokenize(query_str, True), self.bm25_corpus, n=200)
        # bm25_q_list = [' '.join(sent) for sent in bm25_ranked_list]
        # bm25_preds_onlyq = self.question_bank.set_index('tokenized_question_str').loc[bm25_q_list, 'question'].tolist()
        #
        # insert_qs = [_q for _q in bm25_preds if _q in bm25_preds_onlyq]
        # other_qs = [_q for _q in bm25_preds if _q not in bm25_preds_onlyq]

        new_preds = []
        for q in bm25_preds:
            new_preds.append(q)
        for q in self.notin_train_questions[:topn]:
            if q not in bm25_preds:
                new_preds.append(q)
        new_preds = new_preds[:topn]
        return new_preds

    def _get_candidate_scores(self, query_str, candidate_answers, threshold=0.3):
        clariq_pair_datas = []
        for question in candidate_answers:
            clariq_pair_datas.append((query_str, question if question else 'no_q'))

        input_ids, attention_mask, token_type_ids = self._format_bert_datas(self.tokenizer_for_clariq, clariq_pair_datas)
        input_ids_multitask, attention_mask_multitask, token_type_ids_multitask = self._format_bert_datas(self.tokenizer_for_clariq, clariq_pair_datas, 'electra_multitask')

        all_probs = []
        with torch.no_grad():
            n_steps = int(np.ceil(len(input_ids) / self.batch_size))
            for i in range(n_steps):
                start = i * self.batch_size
                end = start + self.batch_size
                probs1 = self._get_model_output_probs(
                    self.model_for_clariq,
                    input_ids[start:end],
                    attention_mask[start:end],
                    token_type_ids[start:end]
                )
                probs2 = self._get_model_output_probs(
                        self.multitask_model_for_clariq,
                        input_ids_multitask[start:end],
                        attention_mask_multitask[start:end],
                        token_type_ids_multitask[start:end]

                )
                probs = probs1 + probs2
                probs = probs.cpu().numpy().tolist()
                all_probs.extend(probs)

        for i in range(len(clariq_pair_datas)):
            if clariq_pair_datas[i][1] == 'no_q':
                all_probs[i] *= 0.9
        filter_answers = []
        sorted_idxs = np.argsort(all_probs)[::-1]
        # for i in range(10):
        #     print(i, all_probs[sorted_idxs[i]], clariq_pair_datas[sorted_idxs[i]][1])
        for idx in sorted_idxs:
            if all_probs[idx] < threshold:
                break
            filter_answers.append((clariq_pair_datas[idx][1], all_probs[idx]))

        return filter_answers

    def _get_answer_classification(self, initial_request, question, answer):
        #  input_ids, attention_mask, token_type_ids = self._format_roberta_datas(self.tokenizer_for_answer, [(answer, None)])
        input_sentence = initial_request + question + answer
        input_ids, attention_mask, token_type_ids = self._format_bert_datas(self.tokenizer_for_answer, [(input_sentence,)])
        probs = self._get_model_output_probs(
            self.model_for_answer,
            input_ids,
            attention_mask,
            token_type_ids
        )
        prob = probs[0]
        prob = prob.cpu().numpy().tolist()
        pred = np.argmax(prob, axis=-1)
        return pred

    def _get_next_question(self, input_data):
        if len(input_data['conversation_context']) < 1:
            need_answer = True
            answer_pred = 0
        else:
            last_uttr = input_data['conversation_context'][-1]
            answer_pred = self._get_answer_classification(input_data['initial_request'], last_uttr['question'], last_uttr['answer'])
            need_answer = answer_pred == 0
        if not need_answer:
            # 不需要再澄清了
            return '', answer_pred
        else:
            asked_questions = set([_u['question'] for _u in input_data['conversation_context']])

            query_str = input_data['initial_request']
            for ctx in input_data['conversation_context']:
                if 'i cannot answer' in ctx['answer'].lower():
                    continue
                clean_answer = re.sub(r'(^|[^a-z])(yes|no)($|[^a-z])', '', ctx['answer'].lower()).strip()
                if len(clean_answer) < 1:
                    if 'yes' in clean_answer:
                        query_str += ' ' + ctx['question'].lower().strip()
                else:
                    query_str += ' ' + clean_answer

            candidate_answers = self._get_new_bm25_candidate_answer(query_str, topn=200)
            filter_answers = self._get_candidate_scores(query_str, candidate_answers, threshold=0.3)
            best_answer = ''
            for ans in filter_answers:
                if ans[0] in asked_questions:
                    continue
                else:
                    best_answer = ans[0]
                    break
            return best_answer, answer_pred

    def _format_outputs(self, input_datas):
        results = []
        n_samples = 0
        n_total = len(input_datas)
        sstime = time.time()
        stime = sstime
        for _, input_data in input_datas.items():
            best_question, answer_pred = self._get_next_question(input_data)
            # print(input_data['initial_request'], best_question, answer_pred)
            res = [input_data['context_id'], 0, '"{}"'.format(best_question if best_question != 'no_q' else ''), answer_pred, 0, 1, 'NTES_ALONG']
            results.append(res)
            n_samples += 1
            if n_samples == 1:
                etime = time.time()
                print('finish: {} / {}, total cost {:.4f}s, {:.4f}s/it'.format(n_samples, n_total, etime - sstime, (etime - stime) / n_samples))
            if n_samples % 20 == 0:
                etime = time.time()
                print('finish: {} / {}, total cost {:.4f}s, {:.4f}s/it'.format(n_samples, n_total, etime - sstime, (etime - stime) / n_samples))
                stime = time.time()
        etime = time.time()
        print('finish: {}, totla cost {:.4f}s, {:.4f}s/it'.format(n_total, etime - sstime, (etime - sstime) / n_total))
        return results

    def handle_inputs(self, input_file_path, output_file_path):
        input_datas = pickle.load(open(input_file_path, 'rb'))
        # input_datas = json.load(open(input_file_path, 'r', encoding='utf-8'))
        results = self._format_outputs(input_datas)
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            for res in results:
                fout.write(' '.join([str(_d) for _d in res]) + '\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Input arguments for ClariQ Stage2 Task.',
        add_help=True
    )
    parser.add_argument('--input_file_path',
                        type=str,
                        help='multi turn input data file path',
                        required=True)
    parser.add_argument('--output_file_path',
                        type=str,
                        help='multi turn result output file path',
                        required=True)
    args = parser.parse_args()

    # initial interface
    interface_config = Config()

    nltk.data.path.append(interface_config.nltk_data_dir)

    interface = Interface(interface_config)

    interface.handle_inputs(args.input_file_path, args.output_file_path)
