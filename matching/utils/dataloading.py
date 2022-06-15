# TODO: add docstring for module
from typing import Any, Dict, List, Tuple

NegAliasesScores = Tuple[List[str], List[Any]]
PositiveAlias_NegativesAliasesScores = List[Tuple[str, str, NegAliasesScores]]
NegativeScoreDict = Dict[str, Dict[str, str]]


def get_aliases_from_line(line: str, is_lowercase: bool):
    items = line[:-1].split('\t')
    if len(items) == 5:
        entity_name, alias1, alias2, neg_alias, _ = items
    else:
        entity_name, alias1, alias2, neg_alias = items

    if len(alias1) >= 1 and len(alias2) >= 1:
        if is_lowercase:
            alias1 = alias1.lower()
            alias2 = alias2.lower()
            neg_alias = neg_alias.lower()
    return alias1, alias2, neg_alias


# Are negative scores required for training?
def load_train_negative_scores(file_path: str) -> NegativeScoreDict:
    score_dict = dict()
    if file_path is None:
        return score_dict
    score_lines = open(file_path, 'r').readlines()
    for line in score_lines:
        entity_name, neg_aliases, neg_scores = line[:-1].split('\t')
        score_dict[entity_name] = {'neg': neg_aliases, 'neg_score': neg_scores}
    return score_dict


def load_data(filename: str, is_lowercase: bool) -> PositiveAlias_NegativesAliasesScores:
    """
    Dataloader for dev and test sets.
    """
    data = []
    kb_link: str
    alias1: str
    alias2: str
    neg_alias: str
    neg_aliases: List[str]
    scores: List[Any] = []
    neg_aliases_scores: NegAliasesScores  # scores is empty list, no pre_negscore file loaded

    for line in open(filename, 'r').readlines():
        alias1, alias2, neg_alias = get_aliases_from_line(line, is_lowercase)
        neg_aliases = neg_alias.split('___')
        neg_aliases_scores = neg_aliases, scores
        data.append((alias1, alias2, neg_aliases_scores))
    return data


def load_data_train(filename: str, is_lowercase: bool, pre_negscore: str) -> PositiveAlias_NegativesAliasesScores:
    data = []

    for line in open(filename, 'r').readlines():
        # Get entity_name pos_alias neg_alias1___neg_alias2___neg_aliasX...
        alias1, alias2, neg_alias = get_aliases_from_line(line, is_lowercase)

        # If negative score of positive/negative aliases are available, construct score dictionary
        # and append data with scores
        neg_score_dict = load_train_negative_scores(pre_negscore)
        if len(neg_score_dict) > 0:  # pre_negscore was given a valid filepath
            if alias1 in neg_score_dict:
                neg_aliases = neg_score_dict[alias1]['neg'].split('__')
                if len(neg_aliases) > 20:
                    neg_scores = neg_score_dict[alias1]['neg_score'].split('__')
                    neg_aliases_scores = neg_aliases, neg_scores
                    data.append((alias1, alias2, neg_aliases_scores))
        else:
            neg_aliases = neg_alias.split('___')
            if len(neg_aliases) > 20:
                neg_aliases_scores = neg_aliases, []
                data.append((alias1, alias2, neg_aliases_scores))
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
