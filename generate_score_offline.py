from dataset import *
from model import *
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
import eval_metrics as em
from sklearn.metrics import (auc, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from typing import Callable, Dict, List, Optional, Tuple, Union
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np


def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/try/')
    parser.add_argument('-s', '--score_dir', type=str, help="folder path for writing score",
                        default='./scores')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would liek to score on",
                        required=False, default='19eval')
    parser.add_argument("--gpu", type=str, help="GPU index", default="6")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.out_score_dir = "./scores"
    return args


def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer


def test_on_19PS(task, feat_model_path, loss_model_path, output_score_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(feat_model_path)
    test_set = ASVspoof2019PS('/home/xieyuankun/data/asv2019PS/preprocess_xls-r-300m', 'eval',
                              'xls-r-300m', feat_len=1050, pad_chop=True, padding='zero')

    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    model.eval()

    txt_file_name = os.path.join(output_score_path, model_name + '_' + task + '_score.txt')

    with open(txt_file_name, 'w') as cm_score_file:
        y_pred = np.array([])
        y = np.array([])
        for i, data_slice in enumerate(tqdm(testDataLoader)):
            w2v2, audio_fn,lenOri, labels = data_slice[0], data_slice[1], data_slice[2], data_slice[3]
            w2v2 = w2v2.transpose(1, 2).to(device)
            labels = labels.to(device)
            # w2v2 = w2v2.squeeze(dim = 1)
            embedding, w2v2_outputs = model(w2v2) 
            score = w2v2_outputs
            score = score.squeeze(dim=0)[:int(lenOri)].cpu() # before calculate EER, delete the padding area according to the lenori.
            score = score.detach().numpy()
            labels = labels.squeeze(dim=0)[:int(lenOri)].cpu() # before calculate EER, delete the padding area according to the lenori.
            labels = labels.detach().numpy()
            y_pred = np.append(y_pred, score, axis=0) # all predict frames 
            y = np.append(y, labels, axis=0) # all label frames
        np.save("./scores/final_label",y)
        np.save("./scores/final_pred",y_pred)
        EER = calculate_eer(y, y_pred)
        print(EER)


if __name__ == "__main__":
    args = init()
    model_dir = os.path.join(args.model_folder)
    model_path = os.path.join(model_dir, "anti-spoofing_feat_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    test_on_19PS(args.task, model_path, loss_model_path, args.score_dir, args.model_name)