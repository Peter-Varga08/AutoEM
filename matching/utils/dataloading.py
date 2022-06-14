# TODO: add docstring
from typing import List, Tuple


def load_data(filename: str, is_lowercase: bool) -> List[Tuple[str, str, Tuple[List[str], list]]]:
    """
    Dataloader for dev and test sets.
    """
    data: List[Tuple[str, str, Tuple[List[str], list]]] = []
    kb_link: str
    alias1: str
    alias2: str
    neg_alias: str

    for ln in open(filename, 'r').readlines():
        items = ln[:-1].split('\t')
        if len(items) == 5:
            kb_link, alias1, alias2, neg_alias, _ = items
        else:
            kb_link, alias1, alias2, neg_alias = items
        if len(alias1) <= 1 or len(alias2) <= 1:
            continue
        if is_lowercase:
            alias1 = alias1.lower()
            alias2 = alias2.lower()
            neg_alias = neg_alias.lower()  # string of all negative aliases
        neg_aliases: List[str]
        neg_aliases = neg_alias.split('___')
        neg: Tuple[List[str], List] = neg_aliases, []
        data.append((alias1, alias2, neg))
    return data


def load_data_train(filename: str, is_lowercase: bool, pre_negscore: str):
    if pre_negscore is not None:
        score_ln = open(pre_negscore, 'r').readlines()
        score_dict = dict()
        for ln in score_ln:
            alias, neg_alias, neg_score = ln[:-1].split('\t')
            score_dict[alias] = {'neg': neg_alias, 'neg_score': neg_score}

    data = list()
    for ln in open(filename, 'r').readlines():
        items = ln[:-1].split('\t')
        if len(items) == 5:
            kb_link, alias1, alias2, neg_alias, _ = items
        else:
            kb_link, alias1, alias2, neg_alias = items

        if len(alias1) <= 1 or len(alias2) <= 1:
            continue

        if is_lowercase:
            alias1 = alias1.lower()
            alias2 = alias2.lower()
            neg_alias = neg_alias.lower()

        if pre_negscore is not None:
            if alias1 not in score_dict:
                continue
            neg_aliases = score_dict[alias1]['neg'].split('__')
            if len(neg_aliases) < 20:
                continue
            neg_scores = score_dict[alias1]['neg_score'].split('__')
            neg = neg_aliases, neg_scores
            data.append((alias1, alias2, neg))
        else:
            neg_aliases = neg_alias.split('___')
            if len(neg_aliases) < 20:
                continue
            neg = neg_aliases, list()
            data.append((alias1, alias2, neg))

    return data


def load_words(examples, ngram):
    vocabulary = set()
    UNK = '<unk>'
    PAD = '<pad>'
    vocabulary.add(PAD)
    vocabulary.add(UNK)
    char2ind = {PAD: 0, UNK: 1}
    ind2char = {0: PAD, 1: UNK}
    for alias1, alias2, _ in examples:
        # Add first alias to vocabulary
        for i in range(0, len(alias1) - (ngram - 1), ngram):
            vocabulary.add(alias1[i:i + ngram])
        if ngram == 2:
            if len(alias1) % 2 == 1:
                vocabulary.add(alias1[len(alias1) - 1])

        # Add second alias to vocabulary
        for i in range(0, len(alias2) - (ngram - 1), ngram):
            vocabulary.add(alias2[i:i + ngram])
        if ngram == 2:
            if len(alias2) % 2 == 1:
                vocabulary.add(alias2[len(alias2) - 1])
    vocabulary = sorted(vocabulary)
    for w in vocabulary:
        idx = len(char2ind)
        char2ind[w] = idx
        ind2char[idx] = w
    return vocabulary, char2ind, ind2char
