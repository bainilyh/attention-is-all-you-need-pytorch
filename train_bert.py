'''
This script handles the training process.
'''

import argparse
import math
import time
import dill as pickle

from tqdm import tqdm
import numpy as np
import random
from random import sample
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import BertForPredict
from transformer.Optim import ScheduledOptim

__author__ = "Yu-Hsiang Huang"


class PctSeqDataset(Dataset):
    def __init__(self, mode, data_base_path, label2id=None, test_number=500):
        super(PctSeqDataset, self).__init__()
        self.sentences = []
        self.labels = []
        self.label2id = label2id

        if mode == "train":
            with open(data_base_path, encoding='UTF-8') as f:
                for line in f:
                    line_list = line.rsplit(',', 1)
                    if len(line_list) < 2:
                        continue
                    category_num = line_list[1].strip('"|\n')
                    if category_num not in self.label2id:
                        continue
                    self.sentences.append(line_list[0].strip('"|\n'))
                    _label_id = self.label2id[category_num]
                    self.labels.append(_label_id)
                print(f'训练数据打印前10条, 数据:{self.sentences[:10]}, 标签:{self.labels[:10]}')
        if mode == "test":
            with open(data_base_path, encoding='UTF-8') as f:
                _sentences = []
                for line in f:
                    _sentences.append(line)
            _sentences = sample(_sentences, test_number)
            for line in _sentences:
                line_list = line.rsplit(',', 1)
                if len(line_list) < 2:
                    continue
                category_num = line_list[1].strip('"|\n')
                if category_num not in self.label2id:
                    continue
                self.sentences.append(line_list[0].strip('"|\n'))
                _label_id = self.label2id[category_num]
                self.labels.append(_label_id)
            print(f'测试数据打印前10条, 数据:{self.sentences[:10]}, 标签:{self.labels[:10]}')

    def __getitem__(self, idx):
        sentences = self.sentences[idx]
        labels = self.labels[idx]
        return sentences, labels

    def __len__(self):
        return len(self.sentences)


# def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
#     ''' Apply label smoothing if needed '''
#
#     loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)
#
#     pred = pred.max(1)[1]
#     gold = gold.contiguous().view(-1)
#     non_pad_mask = gold.ne(trg_pad_idx)
#     n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
#     n_word = non_pad_mask.sum().item()
#
#     return loss, n_correct, n_word


# def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
#     ''' Calculate cross entropy loss, apply label smoothing if needed. '''
#
#     gold = gold.contiguous().view(-1)
#
#     if smoothing:
#         eps = 0.1
#         n_class = pred.size(1)
#
#         one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
#         one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
#         log_prb = F.log_softmax(pred, dim=1)
#
#         non_pad_mask = gold.ne(trg_pad_idx)
#         loss = -(one_hot * log_prb).sum(dim=1)
#         loss = loss.masked_select(non_pad_mask).sum()  # average later
#     else:
#         loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
#     return loss


def patch_src(src):
    src = src.transpose(0, 1)
    return src


# def patch_trg(trg,):
#     trg = trg.transpose(0, 1)
#     trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
#     return trg, gold


def train_epoch(model, training_data, optimizer, opt, device, criterion):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, n_all_total, n_all_correct = 0, 0, 0

    desc = '  - (Training)   '
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        # prepare data
        src_seq = patch_src(batch.src).to(device)
        labels = patch_src(batch.trg).to(device)[:, 1]

        # forward
        optimizer.zero_grad()
        output = model(src_seq)
        loss = criterion(output, labels)

        loss.backward()

        optimizer.step_and_update_lr()

        n_correct = (output.argmax(dim=1) == labels).sum().item()
        n_word = len(labels)
        n_all_total += n_word
        n_all_correct += n_correct
        total_loss += loss.item() * n_word


    loss_per = total_loss / n_all_total
    accuracy = n_all_correct / n_all_total
    return loss_per, accuracy


def eval_epoch(model, validation_data, device, opt, criterion):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_all_total, n_all_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            # prepare data
            src_seq = patch_src(batch.src).to(device)
            labels = patch_src(batch.trg).to(device)[:, 1]

            # forward
            output = model(src_seq)
            loss = criterion(output, labels)

            n_correct = (output.argmax(dim=1) == labels).sum().item()
            # note keeping
            n_word = len(labels)
            n_all_total += n_word
            n_all_correct += n_correct
            total_loss += loss.item() * n_word

    loss_per = total_loss / n_all_total
    accuracy = n_all_correct / n_all_total
    return loss_per, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if opt.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt.output_dir, 'tensorboard'))

    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, ' \
              'elapse: {elapse:3.3f} min'.format(
            header=f"({header})", ppl=ppl,
            accu=100 * accu, elapse=(time.time() - start_time) / 60, lr=lr))

    # valid_accus = []
    valid_losses = []
    criterion = nn.CrossEntropyLoss()
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, opt, device, criterion)
        train_ppl = math.exp(min(train_loss, 100))
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt, criterion)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}

        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100 * train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100 * valid_accu))

        if opt.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu * 100, 'val': valid_accu * 100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)


def load_model(opt, device):
    checkpoint = torch.load(opt.model, map_location=device)
    opt = checkpoint['settings']

    model = BertForPredict(
        opt.src_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model


def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_pkl', default=None)  # all-in-1 data pickle or bpe field
    parser.add_argument('-log', default=None)  # add

    parser.add_argument('-train_path', default=None)  # bpe encoded data
    parser.add_argument('-val_path', default=None)  # bpe encoded data

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('-output_dir', type=str, default=None)
    parser.add_argument('-use_tb', action='store_true')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-model', default=None)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print('No experiment result will be saved.')
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print('[Warning] The warmup steps may be not enough.\n' \
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n' \
              'Using smaller batch w/o longer warmup may cause ' \
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if opt.cuda else 'cpu')

    # ========= Loading Dataset =========#


    training_data, validation_data = prepare_dataloaders_from_bpe_files(opt, device)
    print(opt)

    bert = load_model(opt, device)
    # bert = BertForPredict(
    #     opt.src_vocab_size,
    #     src_pad_idx=opt.src_pad_idx,
    #     trg_emb_prj_weight_sharing=opt.proj_share_weight,
    #     d_k=opt.d_k,
    #     d_v=opt.d_v,
    #     d_model=opt.d_model,
    #     d_word_vec=opt.d_word_vec,
    #     d_inner=opt.d_inner_hid,
    #     n_layers=opt.n_layers,
    #     n_head=opt.n_head,
    #     dropout=opt.dropout,
    #     scale_emb_or_prj=opt.scale_emb_or_prj).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(bert.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)

    train(bert, training_data, validation_data, optimizer, device, opt)


def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    MIN_FREQ = 2
    if not opt.embs_share_weight:
        raise

    data = pickle.load(open(opt.data_pkl, 'rb'))
    # MAX_LEN = data['settings'].max_len
    # field = data['vocab']
    MAX_LEN = 120
    field = data
    fields = (field, field)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    train = TranslationDataset(
        fields=fields,
        path=opt.train_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)
    val = TranslationDataset(
        fields=fields,
        path=opt.val_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)
    return train_iterator, val_iterator


def prepare_dataloaders(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))

    opt.max_token_seq_len = data['settings'].max_len
    opt.src_pad_idx = data['vocab']['src'].vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD]

    opt.src_vocab_size = len(data['vocab']['src'].vocab)
    opt.trg_vocab_size = len(data['vocab']['trg'].vocab)

    # ========= Preparing Model =========#
    if opt.embs_share_weight:
        assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
            'To sharing word embedding the src/trg word2idx table shall be the same.'

    fields = {'src': data['vocab']['src'], 'trg': data['vocab']['trg']}

    train = Dataset(examples=data['train'], fields=fields)
    val = Dataset(examples=data['valid'], fields=fields)

    train_iterator = BucketIterator(train, batch_size=batch_size, device=device, train=True)
    val_iterator = BucketIterator(val, batch_size=batch_size, device=device)

    return train_iterator, val_iterator


if __name__ == '__main__':
    main()
