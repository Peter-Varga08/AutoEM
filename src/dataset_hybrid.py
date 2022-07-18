import pdb
import random
from typing import Callable

import torch
from torch.utils.data import Dataset


def vectorize(ex, char2ind, num_neg):
    """
    Create a vectorized representation for each positive and negative alias within each example.
    For example: "Mike Johns" --> vec = [[num, num, num, num], [num, num, num, num, num]]
    """
    alias1, alias2, neg_aliases = ex
    vec_alias1 = list()
    vec_alias2 = list()
    vec_neg_aliases = list()

    for word in alias1.split():
        char_in_word = [char2ind[ch] if ch in char2ind else char2ind['<unk>'] for ch in word]
        vec_alias1.append(char_in_word)
    for word in alias2.split():
        char_in_word = [char2ind[ch] if ch in char2ind else char2ind['<unk>'] for ch in word]
        vec_alias2.append(char_in_word)
    for i, nalias in enumerate(neg_aliases):
        if len(nalias) <= 1:
            continue
        vec_neg = list()
        for word in nalias.split():
            char_in_word = [char2ind[ch] if ch in char2ind else char2ind['<unk>'] for ch in word]
            vec_neg.append(char_in_word)
        if len(vec_neg) > 0:
            vec_neg_aliases.append(vec_neg)
    assert len(vec_neg_aliases) >= num_neg, f"len(vec_neg_aliases): {len(vec_neg_aliases)}"

    return vec_alias1, vec_alias2, vec_neg_aliases


class AliasDataset(Dataset):
    def __init__(self, examples, ind2char, voc, char2ind, neg_num):
        self.examples = examples
        self.ind2char = ind2char
        self.voc = voc
        self.char2ind = char2ind
        self.neg_num = neg_num

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.char2ind, self.neg_num)

    def lengths(self):
        return [(len(alias1), len(alias2)) for alias1, alias2, *_ in self.examples]


def val_batchify(batch):
    x1_word_len, x1_char_len, x2_word_len, x2_char_len = list(), list(), list(), list()
    x3 = list()
    x3_word_mask = list()
    x3_char_mask = list()
    for ex in batch:
        vec_alias1, vec_alias2, neg_aliases, = ex

        x1_word_len.append(len(vec_alias1))
        for word in vec_alias1:
            x1_char_len.append(len(word))
        x2_word_len.append(len(vec_alias2))
        for word in vec_alias2:
            x2_char_len.append(len(word))
        x3_word_len = list()
        x3_char_len = list()

        for neg_alias in neg_aliases:
            x3_word_len.append(len(neg_alias))
            for word in neg_alias:
                x3_char_len.append(len(word))
        neg_v = torch.LongTensor(len(x3_word_len), max(x3_word_len), max(x3_char_len)).zero_()
        neg_word_mask = torch.ByteTensor(len(x3_word_len), max(x3_word_len)).fill_(1)
        neg_char_mask = torch.ByteTensor(len(x3_word_len), max(x3_word_len), max(x3_char_len)).fill_(1)
        for i, neg_alias in enumerate(ex[2]):
            for j, word in enumerate(neg_alias):
                a3 = torch.LongTensor(word)
                neg_v[i, j, :len(word)].copy_(a3)
                neg_char_mask[i, j, :len(word)].fill_(0)
            neg_word_mask[i, :len(neg_alias)].fill_(0)
        x3.append(neg_v)
        x3_word_mask.append(neg_word_mask)
        x3_char_mask.append(neg_char_mask)

    x1 = torch.LongTensor(len(x1_word_len), max(x1_word_len), max(x1_char_len)).zero_()
    x1_word_mask = torch.ByteTensor(len(x1_word_len), max(x1_word_len)).fill_(1)
    x1_char_mask = torch.ByteTensor(len(x1_word_len), max(x1_word_len), max(x1_char_len)).fill_(1)
    x2 = torch.LongTensor(len(x2_word_len), max(x2_word_len), max(x2_char_len)).zero_()
    x2_word_mask = torch.ByteTensor(len(x2_word_len), max(x2_word_len)).fill_(1)
    x2_char_mask = torch.ByteTensor(len(x2_word_len), max(x2_word_len), max(x2_char_len)).fill_(1)

    for i in range(len(x1_word_len)):
        vec_alias1 = batch[i][0]
        for j, word in enumerate(vec_alias1):
            a1 = torch.LongTensor(word)
            x1[i, j, :len(word)].copy_(a1)
            x1_char_mask[i, j, :len(word)].fill_(0)
        x1_word_mask[i, :len(vec_alias1)].fill_(0)

        vec_alias2 = batch[i][1]
        for j, word in enumerate(vec_alias2):
            a2 = torch.LongTensor(word)
            x2[i, j, :len(word)].copy_(a2)
            x2_char_mask[i, j, :len(word)].fill_(0)
        x2_word_mask[i, :len(vec_alias2)].fill_(0)
    return x1, x1_word_mask, x1_char_mask, x2, x2_word_mask, x2_char_mask, x3, x3_word_mask, x3_char_mask


def train_batchify_(num_neg: int) -> Callable:
    def wrapper_func(*args, **kwargs):
        return train_batchify(num_neg, *args, **kwargs)
    return wrapper_func


def train_batchify(num_neg, batch):
    """
    Batchify train examples by
    Return:
    x1: pos_alias1, batch * max(x1_length) * max(char1_length)
    x2: pos_alias2, batch * max(x2_length) * max(char2_length)
    x3: neg_alias, (batch*num_neg) * max(x3_length)
    """
    # len(neg_subsamples) = len(batch) * num_neg
    neg_alias = list()
    x1_word_len, x1_char_len, x2_word_len, x2_char_len, x3_word_len, x3_char_len = [[] for _ in range(6)]

    # Get number of words and characters per word for each example
    for ex in batch:
        vec_alias1, vec_alias2, neg_candidate = ex

        # alias1 lengths
        x1_word_len.append(len(vec_alias1))
        for word in vec_alias1:
            x1_char_len.append(len(word))

        # alias2 lengths
        x2_word_len.append(len(vec_alias2))
        for word in vec_alias2:
            x2_char_len.append(len(word))

        # neg alias lengths
        for i in range(num_neg):
            neg_alias.append(list())
            x3_word_len.append(list())
            x3_char_len.append(list())
        indices = random.sample(range(len(neg_candidate)), num_neg)
        for i, ind in enumerate(indices):
            neg_alias[i].append(neg_candidate[ind])
            x3_word_len[i].append(len(neg_candidate[ind]))
            for word in neg_candidate[ind]:
                x3_char_len[i].append(len(word))

    # Batchify alias1, alias2 and neg_aliases by creating a padded matrix for the raw data, as well as
    # the masks
    x1 = torch.LongTensor(len(x1_word_len), max(x1_word_len), max(x1_char_len)).zero_()
    x1_word_mask = torch.ByteTensor(len(x1_word_len), max(x1_word_len)).fill_(1)
    x1_char_mask = torch.ByteTensor(len(x1_word_len), max(x1_word_len), max(x1_char_len)).fill_(1)
    x2 = torch.LongTensor(len(x2_word_len), max(x2_word_len), max(x2_char_len)).zero_()
    x2_word_mask = torch.ByteTensor(len(x2_word_len), max(x2_word_len)).fill_(1)
    x2_char_mask = torch.ByteTensor(len(x2_word_len), max(x2_word_len), max(x2_char_len)).fill_(1)
    neg3, neg3_word_mask, neg3_char_mask = list(), list(), list()

    for i in range(len(x1_word_len)):
        vec_alias1 = batch[i][0]
        for j, word in enumerate(vec_alias1):
            a1 = torch.LongTensor(word)
            x1[i, j, :len(word)].copy_(a1)
            x1_char_mask[i, j, :len(word)].fill_(0)
        x1_word_mask[i, :len(vec_alias1)].fill_(0)

        vec_alias2 = batch[i][1]
        for j, word in enumerate(vec_alias2):
            a2 = torch.LongTensor(word)
            x2[i, j, :len(word)].copy_(a2)
            x2_char_mask[i, j, :len(word)].fill_(0)
        x2_word_mask[i, :len(vec_alias2)].fill_(0)

    for j in range(num_neg):
        x3 = torch.LongTensor(len(x3_word_len[j]), max(x3_word_len[j]), max(x3_char_len[j])).zero_()
        x3_word_mask = torch.ByteTensor(len(x3_word_len[j]), max(x3_word_len[j])).fill_(1)
        x3_char_mask = torch.ByteTensor(len(x3_word_len[j]), max(x3_word_len[j]), max(x3_char_len[j])).fill_(1)
        for i in range(len(neg_alias[j])):
            vec_neg = neg_alias[j][i]
            for k, word in enumerate(vec_neg):
                a3 = torch.LongTensor(word)
                x3[i, k, :len(word)].copy_(a3)
                x3_char_mask[i, k, :len(word)].fill_(0)
            x3_word_mask[i, :len(vec_neg)].fill_(0)
        neg3.append(x3)
        neg3_word_mask.append(x3_word_mask.bool())
        neg3_char_mask.append(x3_char_mask.bool())

    return x1, x1_word_mask.bool(), x1_char_mask.bool(), x2, x2_word_mask.bool(), x2_char_mask.bool(), \
           neg3, neg3_word_mask, neg3_char_mask
