''' Translate input text with trained model. '''

import torch
from torch import nn
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from transformer.Models import BertForPredict
from torchtext.data import BucketIterator
from torchtext.datasets import TranslationDataset

import os
import time
import math


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


def prepare_dataloaders_from_bpe_files(opt, device):
    batch_size = opt.batch_size
    data = pickle.load(open(opt.data_pkl, 'rb'))
    MAX_LEN = 120
    field = data
    fields = (field, field)

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

    test = TranslationDataset(
        fields=fields,
        path=opt.test_path,
        exts=('.src', '.trg'),
        filter_pred=filter_examples_with_length)

    opt.max_token_seq_len = MAX_LEN + 2
    opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
    opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)

    test_iterator = BucketIterator(test, batch_size=batch_size, device=device)
    return test_iterator


def patch_src(src):
    src = src.transpose(0, 1)
    return src


def test_epoch(model, validation_data, device, criterion):
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


def test(model, test_data, device, opt):
    ''' Start training '''

    log_test_file = os.path.join(opt.output_dir, 'test.log')

    print('[Info] Training performance will be written to file: {}'.format(log_test_file))

    with open(log_test_file, 'w') as log_vf:
        log_vf.write('epoch,loss,ppl,accuracy\n')

    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr:   {lr:8.5f}, ' \
              'elapse: {elapse:3.3f} min'.format(
            header=f"({header})", ppl=ppl,
            accu=100 * accu, elapse=(time.time() - start_time) / 60, lr=lr))

    test_losses = []
    criterion = nn.CrossEntropyLoss()
    print('[ Epoch', 1, ']')

    start = time.time()
    test_loss, test_accu = test_epoch(model, test_data, device, criterion)
    valid_ppl = math.exp(min(test_loss, 100))
    lr = 0.001
    print_performances('Validation', valid_ppl, test_accu, start, lr)

    with open(log_test_file, 'a') as log_vf:
        log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
            epoch=1, loss=test_loss,
            ppl=valid_ppl, accu=100 * test_accu))



def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate_for_test.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-test_path', default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    parser.add_argument('-log', default=None)  # add
    parser.add_argument('-output_dir', type=str, default=None)


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    device = torch.device('cuda' if opt.cuda else 'cpu')
    testing_data = prepare_dataloaders_from_bpe_files(opt, device)
    bert = load_model(opt, device)
    test(bert, testing_data, device, opt)
    print('[Info] Finished.')


if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()
