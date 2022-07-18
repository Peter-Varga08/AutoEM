import argparse
import logging
import os.path
import time

import torch
from dotenv import load_dotenv
from rich.logging import RichHandler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import dataset_hybrid
import wandb
from model.rnn_encoder import Hybrid_Alias_Sim
from utils.dataloading import load_adg_data, load_data, load_words
from utils.metrics import mrr_score, precision_recall

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True)],
)
LOGGER = logging.getLogger(__name__)

# load environment variable(s)
load_dotenv()
PREV_RUN_COUNT = int(os.getenv('RUN_COUNT'))
CURRENT_RUN_COUNT = PREV_RUN_COUNT + 1
with open('../.env', 'w') as f:
    f.write(f"RUN_COUNT={CURRENT_RUN_COUNT}")


def train(args, data_loader, val_train_loader, model, device, best_mrr):
    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()
    check_point = 100

    for idx, batch in enumerate(tqdm(data_loader)):
        alias1 = batch[0].to(device)
        alias1_word_mask = batch[1].to(device)
        alias1_char_mask = batch[2].to(device)
        alias2 = batch[3].to(device)
        alias2_word_mask = batch[4].to(device)
        alias2_char_mask = batch[5].to(device)
        # 1) calculate positive match loss
        pos_score = model(alias1, alias1_word_mask, alias1_char_mask, alias2, alias2_word_mask, alias2_char_mask)
        pos_score = pos_score.sigmoid().log().sum()
        loss = pos_score

        # 2) calculate negative match losses
        for i in range(args.num_neg):  # iterate over all negative aliases
            neg_alias = batch[6][i].to(device)
            neg_word_mask = batch[7][i].to(device)
            neg_char_mask = batch[8][i].to(device)
            neg_score = model(alias1, alias1_word_mask, alias1_char_mask, neg_alias, neg_word_mask, neg_char_mask)
            neg_score = neg_score.neg().sigmoid().log().sum()
            loss += neg_score

        # normalize loss by number of words/tokens in alias1
        loss = -loss.sum() / alias1.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clip_grad_norm_(model.parameters(), 5)
        print_loss_total += loss.data.cpu().numpy()
        epoch_loss_total += loss.data.cpu().numpy()

        if idx % check_point == 0 and idx > 0:
            print_loss_total = print_loss_total
            print_loss_avg = print_loss_total / check_point

            LOGGER.info('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time() - start))
            print_loss_total = 0
            # compute MRR score on ENTIRE validation dataloader
            score = mrr_score(val_train_loader, model, device)
            LOGGER.info("VALIDATION MRR SCORE IS %.5f:" % score)

            if score > best_mrr:
                LOGGER.info("Saving current model weights...")
                torch.save(model, f"{args.save_model}model_NumNeg{args.num_neg}_BatchSize{args.batch_size}.pt")
                best_mrr = score
            model.train()
            torch.cuda.empty_cache()

    LOGGER.info('epoch loss is: %.5f' % epoch_loss_total)
    return best_mrr


def main() -> None:
    wandb.init(project="AutoEM", entity="petervarga", name=f"RUN_{CURRENT_RUN_COUNT}")

    if not os.path.exists('model'):
        os.mkdir('model')
    if not os.path.exists('log'):
        os.mkdir('log')

    parser = argparse.ArgumentParser(description='Alias Similarity')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--data-workers', type=int, default=4)
    parser.add_argument('--train-file', type=str,
                        default='../artifacts/adg_data_sample:v3/ADGDataSample_international_train.json')
    parser.add_argument('--dev-file', type=str,
                        default='../artifacts/adg_data_sample:v3/ADGDataSample_international_dev.json')
    parser.add_argument('--test-file', type=str,
                        default='../artifacts/adg_data_sample:v3/ADGDataSample_international_test.json')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--input-size', type=int, default=300)
    parser.add_argument('--hidden-size', type=int, default=300)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--embedding-dim', type=int, default=300)
    parser.add_argument('--bidirect', action='store_true', default=True)
    parser.add_argument('--num_neg', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--n-gram', type=int, default=1)
    parser.add_argument('--transfer', action='store_true', default=False)
    parser.add_argument('--base-model', type=str, default='../model/model.pt')
    parser.add_argument('--save-model', type=str, default='../model/')
    parser.add_argument('--load-model', type=str, default=None)
    # parser.add_argument('--lowercase', action='store_true', default=False)
    parser.add_argument('--self-attn', action='store_true', default=True)
    # parser.add_argument('--log-file', type=str, default='../log/log_file.log')
    parser.add_argument('--pre-negscore', type=str, default=None)

    args = parser.parse_args()
    wandb.config.update(args)

    # File logging
    if args.test:
        logfile = logging.FileHandler('../log/log_file_test.log', 'a')
    else:
        logfile = logging.FileHandler('../log/log_file_train.log', 'a')
    file_fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    logfile.setFormatter(file_fmt)
    LOGGER.addHandler(logfile)

    LOGGER.info(f"Executing script with RUN_COUNT: [{CURRENT_RUN_COUNT}].")

    vocab_path_pt = '../model/vocab.pt'
    if args.resume or args.test:  # both for resuming learning and just for loading to test
        LOGGER.info(f'Using previous model: [{os.path.basename(args.load_model)}]')
        model = torch.load(args.load_model)
        voc, char2ind, ind2char = torch.load(vocab_path_pt)['vocabulary']
    else:
        LOGGER.info("LOADING ADG TRAIN AND DEV DATASETS...")
        train_exs = load_adg_data(args.train_file, args.num_neg)
        dev_exs = load_adg_data(args.dev_file, args.num_neg)
        voc, char2ind, ind2char = load_words(train_exs + dev_exs, args.n_gram)
        vocab_dict = voc, char2ind, ind2char
        if not os.path.exists(vocab_path_pt):
            LOGGER.info(
                f"Saving vocabulary of n-gram [{args.n_gram}], created from "
                f"\ntrain set:[{args.train_file}]"
                f"\ndev set: [{args.dev_file}]."
            )
            torch.save({"vocabulary": vocab_dict}, vocab_path_pt)
        model = Hybrid_Alias_Sim(args, voc)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    model.to(device)

    if args.test:
        LOGGER.info(f"Initializing test run with train set: '{args.test_file}'.")
        test_exs = load_adg_data(args.test_file, args.num_neg)
        test_dataset = dataset_hybrid.AliasDataset(test_exs, ind2char, voc, char2ind, args.num_neg)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  sampler=test_sampler, num_workers=args.data_workers,
                                                  collate_fn=dataset_hybrid.val_batchify, pin_memory=args.cuda)
        precision_recall(args, test_loader, model, device, ind2char, LOGGER)
        # error_analysis(args, test_loader, model, device, ind2char)
        LOGGER.info("Test execution has successfully completed.")
        exit()

    # train_vec_rep = train_vec(train_exs, char2ind)
    # train_vec_rep = pickle.load(open('train_vec.pt', 'rb'))

    train_dataset = dataset_hybrid.AliasDataset(train_exs, ind2char, voc, char2ind, args.num_neg)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    dev_dataset = dataset_hybrid.AliasDataset(dev_exs[:1000], ind2char, voc, char2ind, args.num_neg)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                             sampler=dev_sampler, num_workers=args.data_workers,
                                             collate_fn=dataset_hybrid.val_batchify, pin_memory=args.cuda)

    start_epoch = 0
    LOGGER.info('Start training:')
    best_mrr = 0

    for epoch in range(start_epoch, args.num_epochs):
        LOGGER.info('start epoch:%d' % epoch)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   sampler=train_sampler, num_workers=args.data_workers,
                                                   collate_fn=dataset_hybrid.train_batchify_(args.num_neg),
                                                   pin_memory=args.cuda)
        best_mrr = train(args, train_loader, dev_loader, model, device, best_mrr)
    LOGGER.info("Training execution has successfully completed.")


if __name__ == "__main__":
    main()
