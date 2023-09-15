import torch
import torch.nn as nn
import argparse
import os
import json
import shutil
import numpy as np
from model import *
from dataset import *
from torch.utils.data import DataLoader
import torch.utils.data.sampler as torch_sampler
from loss import *
from collections import defaultdict
from tqdm import tqdm, trange
import random
from utils import *
import torch.nn.functional as F
torch.set_default_tensor_type(torch.FloatTensor)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help="random number seed", default=888)
    # Data folder prepare
    parser.add_argument("-a", "--access_type", default='LA')
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path",
                        default='/home/xieyuankun/data/asv2019')
    parser.add_argument("-f", "--path_to_features", type=str, help="features path",
                        default='/home/xieyuankun/data/asv2019PS/preprocess_xls-r-300m')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/try/')
    parser.add_argument("--feat", type=str, help="which feature to use", default='xls-r-300m',)
    parser.add_argument("--feat_len", type=int, help="features length", default=1050)
    parser.add_argument("--lam", type=int, help="weight for emb", default=0.1)
    parser.add_argument('--pad_chop', type=str2bool, nargs='?', const=True, default=True,
                        help="whether pad_chop in the dataset")
    parser.add_argument('--padding', type=str, default='zero', choices=['zero', 'repeat', 'silence'],
                        help="how to pad short utterance")
    parser.add_argument('-m', '--model', help='Model arch', default='TDL',)
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="decay learning rate")
    parser.add_argument('--interval', type=int, default=5, help="interval to decay lr")
    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="13")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")
    parser.add_argument('--base_loss', type=str, default="bce", choices=["ce", "bce"],
                        help="use which loss for basic training")
    parser.add_argument('--add_loss', type=str, default=None,
                        help="add other loss for one-class training")
    parser.add_argument('--test_only', action='store_true',
                        help="test the trained model in case the test crash sometimes or another test method")
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)
    if args.test_only or args.continue_training:
        pass
    else:
        # Path for output data
        if not os.path.exists(args.out_fold):
            os.makedirs(args.out_fold)
        else:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)
        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        # assert os.path.exists(args.path_to_database)
        assert os.path.exists(args.path_to_features)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))

        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def shuffle(feat, ori_len, labels):
    shuffle_index = torch.randperm(labels.shape[0])
    feat = feat[shuffle_index]
    ori_len = ori_len[shuffle_index]
    labels = labels[shuffle_index]
    return feat, ori_len, labels

def cls_loss(scores, labels):
    '''
    calculate classification loss
    1. dispose label, ensure the sum is 1
    2. calculate topk mean, indicates classification score
    3. calculate loss
    '''
    labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-10)
    clsloss = -torch.mean(torch.sum(labels * F.log_softmax(scores, dim=1), dim=1), dim=0)
    return clsloss

def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize model
    if args.model == 'TDL':
        feat_model = TDL().to(args.device)
    emb_loss = EmbeddingLoss()
    if args.continue_training:
        feat_model = torch.load(os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')).to(args.device)
    # feat_model = nn.DataParallel(feat_model, list(range(torch.cuda.device_count())))  # for multiple GPUs
    feat_optimizer = torch.optim.Adam(feat_model.parameters(), lr=args.lr,
                                      betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=0.0005)
    training_set = ASVspoof2019PS(args.path_to_features, 'train',
                                  args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)
    validation_set = ASVspoof2019PS(args.path_to_features, 'dev',
                                    args.feat, feat_len=args.feat_len, pad_chop=args.pad_chop, padding=args.padding)

    trainOriDataLoader = DataLoader(training_set, batch_size=int(args.batch_size),
                                    shuffle=False, num_workers=args.num_workers, collate_fn=training_set.collate_fn,
                                    sampler=torch_sampler.SubsetRandomSampler(range(25380)))
    trainOri_flow = iter(trainOriDataLoader)

    valOriDataLoader = DataLoader(validation_set, batch_size=int(args.batch_size),
                                  shuffle=False, num_workers=args.num_workers, collate_fn=validation_set.collate_fn,
                                  sampler=torch_sampler.SubsetRandomSampler(range(24844)))
    valOri_flow = iter(valOriDataLoader)
    weight = torch.FloatTensor([0.01, 0.99]).to(args.device)
    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        pass
        # criterion = nn.functional.binary_cross_entropy()

    early_stop_cnt = 0
    prev_loss = 1e8
    if args.add_loss is None:
        monitor_loss = 'base_loss'
    else:
        monitor_loss = args.add_loss

    for epoch_num in tqdm(range(args.num_epochs)):
        feat_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        correct_m, total_m, correct_c, total_c, correct_v, total_v = 0, 0, 0, 0, 0, 0
        for i in trange(0, len(trainOriDataLoader), total=len(trainOriDataLoader), initial=0):
            try:
                featOri, audio_fnOri, lenOri, labelsOri = next(trainOri_flow)
            except StopIteration:
                trainOri_flow = iter(trainOriDataLoader)
                featOri, audio_fnOri,lenOri,labelsOri = next(trainOri_flow)
            feat = featOri
            length = lenOri
            labels = labelsOri

            feat = feat.transpose(1, 2).to(args.device)

            length = length.to(args.device)
            labels = labels.to(args.device)
            feat, length, labels = shuffle(feat,length,labels)
            embedding, feat_outputs = feat_model(feat)

            if args.base_loss == "bce":
                BCE_loss = nn.functional.binary_cross_entropy(feat_outputs, labels.float())
                embedding_loss = emb_loss(embedding,length,labels.float())
                feat_loss = BCE_loss + args.lam * embedding_loss
            else:
                CE_loss = cls_loss(feat_outputs, labels)
                embedding_loss = emb_loss(embedding, labels)
                feat_loss = CE_loss + embedding_loss
            trainlossDict['base_loss'].append(feat_loss.item())

            if args.add_loss == None:
                feat_optimizer.zero_grad()
                feat_loss.backward()
                feat_optimizer.step()
            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                          str(trainlossDict[monitor_loss][-1]) + "\n")

        feat_model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            for i in trange(0, len(valOriDataLoader), total=len(valOriDataLoader), initial=0):
                try:
                    featOri, audio_fnOri,lenOri, labelsOri = next(valOri_flow)
                except StopIteration:
                    valOri_flow = iter(valOriDataLoader)
                    featOri, audio_fnOri,lenOri, labelsOri = next(valOri_flow)

                feat = featOri
                length = lenOri
                labels = labelsOri
                feat = feat.transpose(1, 2).to(args.device)
                labels = labels.to(args.device)
                feat, length, labels = shuffle(feat,length,labels)
                embedding, feat_outputs= feat_model(feat)
                if args.base_loss == "bce":
                    BCE_loss = nn.functional.binary_cross_entropy(feat_outputs, labels.float())
                    embedding_loss = emb_loss(embedding,length,labels)
                    feat_loss = BCE_loss + args.lam * embedding_loss
                    score = feat_outputs[:, 0]
                else:
                    CE_loss = cls_loss(feat_outputs, labels)
                    embedding_loss = emb_loss(embedding, labels)
                    feat_loss = CE_loss + embedding_loss
                    score = feat_outputs[:, 0]
                idx_loader.append((labels))
                if args.add_loss in [None]:
                    devlossDict["base_loss"].append(feat_loss.item())
                score_loader.append(score)
                desc_str = ''
                for key in sorted(devlossDict.keys()):
                    desc_str += key + ':%.5f' % (np.nanmean(devlossDict[key])) + ', '
                with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                    log.write(str(epoch_num) + "\t" +
                              str(np.nanmean(devlossDict[monitor_loss])) +
                              "\n")
        valLoss = np.nanmean(devlossDict[monitor_loss])
        if (epoch_num + 1) % 1 == 0:
            torch.save(feat_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))
            loss_model = None

        if valLoss < prev_loss:
            # Save the model checkpoint
            torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            loss_model = None
            prev_loss = valLoss
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 500:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 499))
            break
    return feat_model, loss_model

if __name__ == "__main__":
    args = initParams()
    if not args.test_only:
        _, _ = train(args)

