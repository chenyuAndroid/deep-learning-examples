import torch
import torch.nn as nn  #神经网络工具箱torch.nn
import torch.nn.functional as F  #神经网络函数torch.nn.functional
import numpy as np
import sys
import math

class GloveModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        #声明v和w为Embedding向量
        self.v = nn.Embedding(vocab_size, embed_size)
        self.w = nn.Embedding(vocab_size, embed_size)
        self.biasv = nn.Embedding(vocab_size, 1)
        self.biasw = nn.Embedding(vocab_size, 1)

        #随机初始化参数
        initrange = 0.5 / self.embed_size
        self.v.weight.data.uniform_(-initrange, initrange)
        self.w.weight.data.uniform_(-initrange, initrange)

    def forward(self, i, j, co_occur, weight):
        vi = self.v(i)
        wj = self.w(j)
        bi = self.biasv(i)
        bj = self.biasw(j)

        similarity = torch.mul(vi, wj)
        similarity = torch.sum(similarity, dim=1)

        loss = similarity + bi + bj - torch.log(co_occur)
        loss = 0.5 * weight * loss * loss

        return loss.sum().mean()

    def gloveMatrix(self):
        '''
        获得词向量，这里把两个向量相加作为最后的词向量
        :return:
        '''
        return self.v.weight.data.numpy() + self.w.weight.data.numpy()
