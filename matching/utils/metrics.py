from tqdm import tqdm


def mrr_score(data_loader, model, device, logger):
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
        neg_alias_list = batch[6]
        neg_word_mask_list = batch[7]
        neg_char_mask_list = batch[8]

        pos_scores = model(alias1, alias1_word_mask, alias1_char_mask, alias2, alias2_word_mask, alias2_char_mask)
        pos_scores = pos_scores.data.cpu().numpy().tolist()

        for i in range(len(pos_scores)):
            neg_alias = neg_alias_list[i].to(device)
            neg_word_mask = neg_word_mask_list[i].to(device)
            neg_char_mask = neg_char_mask_list[i].to(device)

            pos_word_len = alias1_word_mask[i].data.cpu().eq(0).long().sum().item()
            pos_char_len = alias1_char_mask[i].data.cpu().eq(0).long().sum(1).numpy().tolist()

            pos_alias2 = alias1[i, :pos_word_len, :max(pos_char_len)].repeat(len(neg_alias), 1, 1)
            pos_word_mask = alias1_word_mask[i, :pos_word_len].repeat(len(neg_alias), 1)
            pos_char_mask = alias1_char_mask[i, :pos_word_len, :max(pos_char_len)].repeat(len(neg_alias), 1, 1)

            neg_scores = model(pos_alias2, pos_word_mask, pos_char_mask, neg_alias, neg_word_mask,
                               neg_char_mask).data.cpu().numpy().tolist()
            pos_score = pos_scores[i]

            # MRR score compute
            neg_scores.append(pos_score)
            sorted_idx = sorted(range(len(neg_scores)), key=neg_scores.__getitem__, reverse=True)
            ranking = ranking + 1 / (sorted_idx.index(len(neg_scores) - 1) + 1)
            num_examples += 1

    ranking /= num_examples
    logger.info("MRR SCORE IS %.5f:" % ranking)
    return ranking


def precision_recall(args, data_loader, model, device, ind2char):
    fp = open('ppl_pr_errors_1.txt', 'w')
    fp1 = open('ppl_pr_errors_0.txt', 'w')
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

            pos_word_len = alias1_word_mask[i].data.cpu().eq(0).long().sum().item()
            pos_char_len = alias1_char_mask[i].data.cpu().eq(0).long().sum(1).numpy().tolist()

            pos_alias2 = alias1[i, :pos_word_len, :max(pos_char_len)].repeat(len(neg_alias), 1, 1)
            pos_word_mask = alias1_word_mask[i, :pos_word_len].repeat(len(neg_alias), 1)
            pos_char_mask = alias1_char_mask[i, :pos_word_len, :max(pos_char_len)].repeat(len(neg_alias), 1, 1)

            neg_scores = model(pos_alias2, pos_word_mask, pos_char_mask, neg_alias, neg_word_mask,
                               neg_char_mask).data.cpu().numpy().tolist()
            pos_score = pos_scores[i]
            pos_score_list.append(pos_score)

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
