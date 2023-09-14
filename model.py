import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
import os
from torch import Tensor
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from pytorch_model_summary import summary
import math


class AttentionConv1d(nn.Module):
    def __init__(self, kernel_size, out_channels):
        super(AttentionConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def calculate_similarity(self, embedding, embedding_neighbor):
        similarity = self.cosine_similarity(embedding, embedding_neighbor)
        similarity = torch.unsqueeze(similarity, dim=1)
        return similarity

    def cal_local_attenttion(self, embedding, feature, kernel_size):
        embedding_l = torch.zeros_like(embedding)
        embedding_l[:, :, 1:] = embedding[:, :, :-1]
        similarity_l = self.calculate_similarity(embedding, embedding_l)
        similarity_c = self.calculate_similarity(embedding, embedding)
        embedding_r = torch.zeros_like(embedding)
        embedding_r[:, :, :-1] = embedding[:, :, 1:]
        similarity_r = self.calculate_similarity(embedding, embedding_r)
        similarity = torch.cat([similarity_l, similarity_c, similarity_r], dim=1)  
        # expand for D times
        batch, channel, temporal_length = feature.size()
        similarity_tile = torch.zeros(batch, kernel_size * channel, temporal_length).type_as(feature)
        similarity_tile[:, :channel * 1, :] = similarity[:, :1, :]
        similarity_tile[:, channel * 1:channel * 2, :] = similarity[:, 1:2, :]
        similarity_tile[:, channel * 2:, :] = similarity[:, 2:, :]
        return similarity_tile

    def forward(self, feature, embedding, weight):
        batch, channel, temporal_length = feature.size()
        inp = torch.unsqueeze(feature, dim=3)
        w = torch.unsqueeze(weight, dim=3)

        unfold = nn.Unfold(kernel_size=(self.kernel_size, 1), stride=1, padding=[1, 0])
        inp_unf = unfold(inp)
        # local attention
        attention = self.cal_local_attenttion(embedding, feature, kernel_size=self.kernel_size)
        inp_weight = inp_unf * attention
        inp_unf_t = inp_weight.transpose(1, 2)
        w_t = w.view(w.size(0), -1).t()
        results = torch.matmul(inp_unf_t, w_t)
        out_unf = results.transpose(1, 2)
        out = out_unf.view(batch, self.out_channels, temporal_length)
        return out


class FilterModule(nn.Module):
    def __init__(self):
        super(FilterModule, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_1_att = AttentionConv1d(kernel_size=3, out_channels=1024)
        self.conv_2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2_att = AttentionConv1d(kernel_size=3, out_channels=1024)
        self.lrelu = nn.LeakyReLU()
        self.drop_out = nn.Dropout(0.7)
    def forward(self, x, embedding):
        feat1 = self.lrelu(self.conv_1_att(x, embedding, self.conv_1.weight))
        feat2 = self.lrelu(self.conv_2_att(feat1, embedding, self.conv_2.weight))
        feature = self.drop_out(feat2)
        return feat1, feature

class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule, self).__init__()
        self.conv = nn.Conv1d(in_channels=1024, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(2100, 132)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        out = self.sig(x)
        return out
class EmbeddingModule(nn.Module):
    def __init__(self):
        super(EmbeddingModule, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU()
    def forward(self, x):
        out = self.lrelu(self.conv_1(x))
        out = self.conv_2(out)
        embedding = F.normalize(out, p=2, dim=1)
        return embedding
class TDL(nn.Module):
    def __init__(self):
        super(TDL, self).__init__()
        self.filter_module = FilterModule()
        self.base_module = BaseModule()
        self.classifier_module = ClassifierModule()
        self.softmax = nn.Softmax(dim=1)
        self.embedding_module = EmbeddingModule()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        #weights = self.filter_module(x)
        #x = weights * x
        embedding = self.embedding_module(x)
        feature1, feature2 = self.base_module(x, embedding)
        feature2 = self.classifier_module(feature2)
        return embedding, feature2


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    #print(summary(Network(), torch.randn((16, 1024, 1050)), show_input=False))

