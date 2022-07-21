from model import UniSkip, Encoder
from data_loader import DataLoader
from vocab import load_dictionary
from config import *
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable
import torch
import re
import numpy as np
import pickle


class UsableTransformer:
    # @profile
    def __init__(self, model_path, vocab_path):
        print("Loading Vocab", vocab_path)
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        # self.vocab = dataset.WordVocab.load_vocab(vocab_path)
        print("Vocab Size: ", len(self.vocab))
        self.model = torch.load(model_path)
        if USE_CUDA:
            self.model.cuda(CUDA_DEVICE)
    # @profile
    def encode(self, text, numpy=True):
        segment_label = []
        sequence = []
        for t in text:
            l = len(t.split(' ')) * [1]
            s = self.vocab.to_seq(t)
            if len(l) > 30:
                segment_label.append(l[:30])
            else:
                segment_label.append(l + [0]*(30-len(l)))
            if len(s) > 30:
                 sequence.append(s[:30])
            else:
                sequence.append(s + [0]*(30-len(s)))

        segment_label = torch.LongTensor(segment_label)
        sequence = torch.LongTensor(sequence)

        if USE_CUDA:
            sequence = sequence.cuda(CUDA_DEVICE)
            segment_label = segment_label.cuda(CUDA_DEVICE)
        
        encoded = self.model.encode(sequence, segment_label)
        result = torch.mean(encoded.detach(), dim=1)

        del encoded
        if USE_CUDA:
            if numpy:
                return result.data.cpu().numpy()
            else:
                return result.to('cpu')
        else:
            if numpy:
                return result.data.numpy()
            else:
                return result
