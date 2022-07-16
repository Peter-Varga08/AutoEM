# TODO: add docstring for module
import json
from typing import Any, Dict, List, Tuple

PositiveAliasX_NegativesAliases = List[Tuple[Any, ...]]
PositiveAlias2_NegativesAliases = List[Tuple[str, str, List[str]]]
NegativeScoreDict = Dict[str, Dict[str, str]]


def get_aliases_from_line(line: str):
    items = line[:-1].split('\t')
    assert all([name.islower() for name in items]), "Dataset lacks lowercase formatting."
    entity_name, alias1, alias2, neg_alias = items
    assert all([len(alias) > 1 for alias in (alias1, alias2)])
    return alias1, alias2, neg_alias


def load_data(filename: str) -> PositiveAlias2_NegativesAliases:
    """
    Dataloader from txt file for train, dev and test sets.
    """
    data = []
    alias1: str
    alias2: str
    neg_alias: str
    neg_aliases: List[str]

    for line in open(filename, 'r').readlines():
        alias1, alias2, neg_alias = get_aliases_from_line(line)
        neg_aliases = neg_alias.split('___')
        data.append((alias1, alias2, neg_aliases))
    return data


def load_adg_data(filename: str, num_pos: int = 2, num_neg: int = 5) -> PositiveAliasX_NegativesAliases:
    data = []
    alias1: str
    alias2: str
    neg_alias: str
    neg_aliases: List[str]

    with open(filename, 'r') as json_file:
        tmp = json.load(json_file)
    for idx in tmp.keys():
        group = tmp[idx]
        kb_link, pos_aliases, neg_aliases = group
        assert num_pos <= len(pos_aliases)
        assert num_neg <= len(neg_aliases)
        pos_aliases = pos_aliases[0:num_pos]
        neg_aliases = neg_aliases[0:num_neg]
        data.append((*pos_aliases, neg_aliases))
    return data


def load_words(examples, ngram):
    """
    Construct a vocabulary of N-grams of the positive aliases.
    The 'vocabulary' set size defines the number of embeddings to be contained within the nn.Embedding layer of the
    'Hybrid Alias Sim' model.
    """
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

# Are negative scores required for training?
# def load_train_negative_scores(file_path: str) -> NegativeScoreDict:
#     score_dict = dict()
#     if file_path is None:
#         return score_dict
#     score_lines = open(file_path, 'r').readlines()
#     for line in score_lines:
#         entity_name, neg_aliases, neg_scores = line[:-1].split('\t')
#         score_dict[entity_name] = {'neg': neg_aliases, 'neg_score': neg_scores}
#     return score_dict
