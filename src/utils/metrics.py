from tqdm import tqdm


def mrr_score(data_loader, model, device):
    """
    Compute normalized MRR score for all batches of a validation dataloader, with the 'correct response' being the
    pos_score of alias1,alias2 in each batch of 1 positive and 5 negative examples.
    """

    model.eval()
    ranking = 0
    num_examples = 0
    for idx, batch in enumerate(tqdm(data_loader)):
        alias1 = batch[0].to(device)
        alias1_word_mask = batch[1].to(device)
        alias1_char_mask = batch[2].to(device)
        alias2 = batch[3].to(device)
        alias2_word_mask = batch[4].to(device)
        alias2_char_mask = batch[5].to(device)
        neg_alias_list = batch[6]  # batch_size(32) x num_neg(5) x word_max_len x char_max_len
        neg_word_mask_list = batch[7]
        neg_char_mask_list = batch[8]

        pos_scores = model(alias1, alias1_word_mask, alias1_char_mask, alias2, alias2_word_mask, alias2_char_mask)
        pos_scores = pos_scores.data.cpu().numpy().tolist()

        for i in range(len(pos_scores)):  # iterate over batch (32 examples) of positive scores, i.e. shape [1,32]
            neg_alias = neg_alias_list[i].to(device)
            neg_word_mask = neg_word_mask_list[i].to(device)
            neg_char_mask = neg_char_mask_list[i].to(device)

            # select pos word and char lens of current batch
            pos_word_len = alias1_word_mask[i].data.cpu().eq(0).long().sum().item()
            pos_char_len = alias1_char_mask[i].data.cpu().eq(0).long().sum(1).numpy().tolist()
            # Create as many duplicates of pos word- and char masks as len(neg_alias)
            num_neg = len(neg_alias)
            pos_alias2 = alias1[i, :pos_word_len, :max(pos_char_len)].repeat(num_neg, 1, 1)
            pos_word_mask = alias1_word_mask[i, :pos_word_len].repeat(num_neg, 1)
            pos_char_mask = alias1_char_mask[i, :pos_word_len, :max(pos_char_len)].repeat(num_neg, 1, 1)

            # calculate num_neg amount of neg scores using num_neg amount of identical pos_alias2
            neg_scores = model(pos_alias2, pos_word_mask, pos_char_mask, neg_alias, neg_word_mask,
                               neg_char_mask).data.cpu().numpy().tolist()
            pos_score = pos_scores[i]

            # MRR score compute: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
            # 1) In each batch, 5 neg scores and 1 pos score constitute neg_score
            # 2) sort neg_scores in decreasing ranked order
            # 3) accumulate ranking value
            # +) Make sure that model(*args, **kwargs) returns neg_scores in
            neg_scores.append(pos_score)  # append pos_score of current batch
            sorted_idx = sorted(range(len(neg_scores)), key=neg_scores.__getitem__, reverse=True)
            # get idx of pos_score element (CANONICAL ELEMENT)...
            # and adjust index number by +1 to allow division with zero, e.g. 1/0 -> 1/1
            # if 'ranking_denominator' is equal to 1, it means that the pos_score has the highest rank (correct output)
            ranking_denominator = sorted_idx.index(len(neg_scores) - 1) + 1
            ranking += 1 / ranking_denominator
            num_examples += 1

    # In order to get highest value of 'ranking', 'ranking_denominator' would need to always be equal to 1
    ranking /= num_examples
    return ranking


def precision_recall(args, data_loader, model, device, ind2char) -> None:
    def unused_author_output() -> None:
        """
        Write output of classification in the form of a character list upon the positive and negative scores surpassing
        an arbitrary threshold. Usage is only recommended for tentative experimental purposes and fine-tunings.
        """
        fp = open('ppl_pr_errors_1.txt', 'w')
        fp1 = open('ppl_pr_errors_0.txt', 'w')
        if pos_score < 0.247:
            ex = alias1[i].data.cpu()
            pos_ex = alias2[i].data.cpu()
            ch_list = list()

            for j, word in enumerate(ex):
                if batch[1][i, j].item() == 1:
                    continue
                for k, ch in enumerate(word):
                    if batch[2][i, j, k].item() == 1:
                        continue
                    else:
                        ch = ch.item()
                        if ch in ind2char:
                            ch_list.append(ind2char[ch])
                        else:
                            ch_list.append(ind2char[1])
                ch_list.append(' ')

            fp1.write(''.join(ch_list[:-1]) + '\t')

            ch_list = list()
            for j, word in enumerate(pos_ex):
                if batch[4][i, j].item() == 1:
                    continue
                for k, ch in enumerate(word):
                    if batch[5][i, j, k].item() == 1:
                        continue
                    else:
                        ch = ch.item()
                        if ch in ind2char:
                            ch_list.append(ind2char[ch])
                        else:
                            ch_list.append(ind2char[1])
                ch_list.append(' ')
            fp1.write(''.join(ch_list[:-1]) + '\t' + str(pos_score) + '\t' + str('1') + '\n')

        for m, v in enumerate(neg_scores):
            neg_score_list.append(v)
            if v > 6.582:
                ex = alias1[i].data.cpu()
                ch_list = list()

                for j, word in enumerate(ex):
                    if batch[1][i, j].item() == 1:
                        continue
                    for k, ch in enumerate(word):
                        if batch[2][i, j, k].item() == 1:
                            continue
                        else:
                            ch = ch.item()
                            if ch in ind2char:
                                ch_list.append(ind2char[ch])
                            else:
                                ch_list.append(ind2char[1])
                    ch_list.append(' ')

                fp.write(''.join(ch_list[:-1]) + '\t')
                ch_list = list()

                neg_ex = neg_alias[m].data.cpu()
                for j, word in enumerate(neg_ex):
                    if neg_word_mask[m, j].item() == 1:
                        continue
                    for k, ch in enumerate(word):
                        if neg_char_mask[m, j, k].item() == 1:
                            continue
                        else:
                            ch = ch.item()
                            if ch in ind2char:
                                ch_list.append(ind2char[ch])
                            else:
                                ch_list.append(ind2char[1])
                    ch_list.append(' ')
                fp.write(''.join(ch_list[:-1]) + '\t' + str(v) + '\t' + str('0') + '\n')

    model.eval()
    ranking = 0
    num_examples = 0
    recall_list = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1,
                   0.05]
    pos_score_list = list()
    neg_score_list = list()

    for idx, batch in enumerate(tqdm(data_loader)):
        alias1 = batch[0].to(device)
        alias1_word_mask = batch[1].to(device)
        alias1_char_mask = batch[2].to(device)
        alias2 = batch[3].to(device)
        alias2_word_mask = batch[4].to(device)
        alias2_char_mask = batch[5].to(device)
        neg_alias_list = batch[6]
        neg_word_mask_list = batch[7]
        neg_char_mask_list = batch[8]

        pos_scores = model(alias1, alias1_word_mask, alias1_char_mask, alias2, alias2_word_mask, alias2_char_mask)
        pos_scores = pos_scores.data.cpu().numpy().tolist()

        for i in range(len(pos_scores)):
            neg_alias = neg_alias_list[i].to(device)
            neg_word_mask = neg_word_mask_list[i].to(device)
            neg_char_mask = neg_char_mask_list[i].to(device)

            # Create as many duplicates of positive match as len(neg_alias)
            num_neg = len(neg_alias)
            pos_word_len = alias1_word_mask[i].data.cpu().eq(0).long().sum().item()
            pos_char_len = alias1_char_mask[i].data.cpu().eq(0).long().sum(1).numpy().tolist()
            pos_alias2 = alias1[i, :pos_word_len, :max(pos_char_len)].repeat(num_neg, 1, 1)
            pos_word_mask = alias1_word_mask[i, :pos_word_len].repeat(num_neg, 1)
            pos_char_mask = alias1_char_mask[i, :pos_word_len, :max(pos_char_len)].repeat(num_neg, 1, 1)

            neg_scores = model(pos_alias2, pos_word_mask, pos_char_mask, neg_alias, neg_word_mask,
                               neg_char_mask).data.cpu().numpy().tolist()
            pos_score = pos_scores[i]
            pos_score_list.append(pos_score)
            unused_author_output()

            neg_scores.append(pos_score)
            sorted_idx = sorted(range(len(neg_scores)), key=neg_scores.__getitem__, reverse=True)
            ranking = ranking + 1 / (sorted_idx.index(len(neg_scores) - 1) + 1)
            num_examples += 1
            # rk = sorted_idx.index(len(neg_scores) - 1)

    pos_sorted_idx = sorted(range(len(pos_score_list)), key=pos_score_list.__getitem__, reverse=True)
    ranking = ranking / num_examples
    print("MRR SCORE IS: %.5f" % ranking)

    for recall in recall_list:
        num_correct_labels = int(num_examples * recall)
        score_limit = pos_score_list[pos_sorted_idx[num_correct_labels]]

        # precision
        lb_list = [i for i in range(len(neg_score_list)) if neg_score_list[i] >= score_limit]
        precision = num_correct_labels / (len(lb_list) + num_correct_labels)
        print('recall is: %.2f,  precision is: %.3f, score is: %.3f' % (recall, precision, score_limit))

    print('\n')
