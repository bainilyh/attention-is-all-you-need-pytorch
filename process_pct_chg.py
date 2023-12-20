#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import random
import math
from random import sample

import codecs


def pct_trans(pct, map_pct_id=None):
    if pct <= -9.7:
        pct = -10.1
    elif pct >= 9.7:
        pct = 10.1
    pct = math.ceil(pct)
    if map_pct_id is None:
        return pct
    return map_pct_id[pct]


def seq_data_iter_random_n(code_chg_list, src_path, trg_path, min_seq=15, max_seq=60):
    corpus_X = []
    for corpus in code_chg_list:
        len_list = len(corpus)
        for i in range(0, len_list, 15):
            num_steps = random.randint(min_seq, max_seq)
            if i + num_steps + 1 > len_list:
                if i + 5 > len_list:
                    continue
                num_steps = len_list - 1 - i
            sub_corpus = corpus[i: i + num_steps + 1]
            corpus_X.append(sub_corpus)
    print(f'一共有{len(corpus_X)}个样本')
    with codecs.open(src_path, 'w', encoding='utf-8') as src_f, codecs.open(trg_path, 'w', encoding='utf-8') as trg_f:
        cntr = 0
        for i in range(len(corpus_X)):
            X = ' '.join(map(str, corpus_X[i][:-1]))
            Y = str(corpus_X[i][-1])
            src_f.write(X + '\n')
            trg_f.write(Y + '\n')
        assert cntr == 0, 'Number of lines in two files are inconsistent.'
    return src_path, trg_path


# %%
with open('./data/pct_chg.plk', 'rb') as f:
    code_chg_list = pickle.load(f)

# 测试的预热序列
test_chg_list = code_chg_list[300][-50:-1]
# 测试预测的真是数值
test_right = code_chg_list[300][-1]

print(len(code_chg_list))

code_chg_list = [[pct_trans(pct) for pct in sub_seq] for sub_seq in code_chg_list]
seq_data_iter_random_n(code_chg_list, './data/test.src', './data/test.trg')

