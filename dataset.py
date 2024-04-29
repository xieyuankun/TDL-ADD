#!/usr/bin/python3

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import librosa
from feature_extraction import LFCC
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_features, part='train', feature='LFCC', feat_len=750, pad_chop=True,
                 padding='repeat', genuine_only=False):
        super(ASVspoof2019, self).__init__()
        self.access_type = access_type
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.genuine_only = genuine_only
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8,
                        "A09": 9,
                        "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17,
                        "A18": 18,
                        "A19": 19}
        elif self.access_type == 'PA':
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        else:
            raise ValueError("Access type should be LA or PA!")
        self.label = {"spoof": 1, "bonafide": 0}
        self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        if self.genuine_only:
            assert self.access_type == "LA"
            if self.part in ["train", "dev"]:
                num_bonafide = {"train": 2580, "dev": 2548}
                self.all_files = self.all_files[:num_bonafide[self.part]]
            else:
                res = []
                for item in self.all_files:
                    if "bonafide" in item:
                        res.append(item)
                self.all_files = res
                assert len(self.all_files) == 7355

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        assert len(all_info) == 6
        featureTensor = torch.load(filepath)

        if self.feature == "wav2vec2_largeraw":
            # featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        # featureTensor = featureTensor.permute(0, 2, 1)
        filename = "_".join(all_info[1:4])
        tag = self.tag[all_info[4]]
        label = self.label[all_info[5]]
        return featureTensor, filename, tag, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [padding_Tensor(sample[0], max_len) for sample in samples]
            max_len_label = max([sample[2].shape[1] for sample in samples])
            filename = [sample[1] for sample in samples]
            label = [sample[2] for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(filename), default_collate(label)


class ASVspoof2019PS(Dataset):
    def __init__(self, path_to_features, part='train', feature='W2V2', feat_len=1050, pad_chop=True, padding='repeat'):
        super(ASVspoof2019PS, self).__init__()
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.feat_len = feat_len
        self.feature = feature
        self.pad_chop = pad_chop
        self.padding = padding
        self.label = {"spoof": 1, "bonafide": 0}
        self.path = os.path.join(self.ptf, 'xls-r-300m')
        # self.all_files = librosa.util.find_files(os.path.join(self.ptf, self.feature), ext="pt")
        protocol = os.path.join(os.path.join('/home/xieyuankun/data/asv2019PS/ASVspoof2019_PS_cm_protocols/',
                                             'PS_' + self.part + '_0.16_real1_pad1.txt'))
        with open(protocol, 'r') as f:
            audio_info = [info.strip().split(' ', 2) for info in f.readlines()]
        self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, ori_len, label = self.all_info[idx]
        filepath = os.path.join(self.path, filename + ".pt")
        basename = os.path.basename(filepath)
        all_info = basename.split(".")[0].split("_")
        featureTensor = torch.load(filepath)
        if self.feature == "wav2vec2_largeraw":
            # featureTensor = featureTensor.permute(0, 2, 1)
            featureTensor = featureTensor.float()
        this_feat_len = featureTensor.shape[1]
        if self.pad_chop:
            if this_feat_len > self.feat_len:
                startp = np.random.randint(this_feat_len - self.feat_len)
                featureTensor = featureTensor[:, startp:startp + self.feat_len, :]
            if this_feat_len < self.feat_len:
                if self.padding == 'zero':
                    featureTensor = padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'repeat':
                    featureTensor = repeat_padding_Tensor(featureTensor, self.feat_len)
                elif self.padding == 'silence':
                    featureTensor = silence_padding_Tensor(featureTensor, self.feat_len)
                else:
                    raise ValueError('Padding should be zero or repeat!')
        else:
            pass
        # featureTensor = featureTensor.permute(0, 2, 1)
        label = eval(label)
        label = torch.tensor(label, dtype=torch.float32)
        ori_len = eval(ori_len)
        ori_len = torch.tensor(ori_len, dtype=torch.float32).unsqueeze(dim=0)
        featureTensor = torch.squeeze(featureTensor,dim=0)
        return featureTensor, filename, ori_len, label

    def collate_fn(self, samples):
        if self.pad_chop:
            return default_collate(samples)
        else:
            # feat_mat = [sample[0].transpose(0,1) for sample in samples]
            # from torch.nn.utils.rnn import pad_sequence
            # feat_mat = pad_sequence(feat_mat, True).transpose(1,2)
            max_len = max([sample[0].shape[1] for sample in samples]) + 1
            feat_mat = [padding_Tensor(sample[0], max_len) for sample in samples]
            filename = [sample[1] for sample in samples]
            max_len_label = max([sample[2].shape[0] for sample in samples])

            label = [pad_tensor(sample[2],max_len_label) for sample in samples]
            # this_len = [sample[3] for sample in samples]

            # return feat_mat, default_collate(tag), default_collate(label), default_collate(this_len)
            return default_collate(feat_mat), default_collate(filename), default_collate(label)

def pad_tensor(a,ref_len):
    tensor = torch.tensor(a)
    padding_length = ref_len - tensor.shape[0]
    padding = torch.zeros(padding_length)
    padded_tensor = torch.cat((tensor, padding))
    return padded_tensor

def padding_Tensor(spec, ref_len):
    _, cur_len, width = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    zero = torch.zeros((1, padd_len, width), dtype=spec.dtype).cuda()
    return torch.cat((spec, zero), 1)


def repeat_padding_Tensor(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul, 1)[:, :ref_len, :]
    return spec


def silence_padding_Tensor(spec, ref_len):
    _, cur_len, width = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((silence_pad_value.repeat(1, padd_len, 1).to(spec.device), spec), 1)


if __name__ == "__main__":
    training_set = ASVspoof2019PS('/home/xieyuankun/data/asv2019PS/preprocess_xls-r-300m', 'train', feature='xls-r-300m'
                                  , feat_len=750, pad_chop=False, padding='zero')
    trainDataLoader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=0,
                                 collate_fn=training_set.collate_fn)
    feat_mat_batch, filename, label = [d for d in next(iter(trainDataLoader))]

    print(feat_mat_batch.shape)
    print(filename)
    print(label.shape)
    # print(feat_mat_batch)
    # feat_mat, filename = training_set[3]
    # print(feat_mat.shape)
