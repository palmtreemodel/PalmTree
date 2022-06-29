import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import *
import numpy as np
import bert_pytorch
from bert_pytorch import dataset
from bert_pytorch import trainer
import pickle as pkl

vocab_path = "data/test_vocab"
train_dataset = "data/training/dfg/temp.txt"
test_dataset = "data/training/dfg/temp.txt"
output_path = "saved_models/transformer"

with open(train_dataset, "r", encoding="utf-8") as f:
    vocab = dataset.WordVocab(f, max_size=VOCAB_SIZE, min_freq=1)

print("VOCAB SIZE:", len(vocab))
vocab.save_vocab(vocab_path)




print("Loading Vocab", vocab_path)
vocab = dataset.WordVocab.load_vocab(vocab_path)
print("Vocab Size: ", len(vocab))
print(vocab.itos)

print("Loading Train Dataset", train_dataset)
train_dataset = dataset.BERTDataset(train_dataset, vocab, seq_len=MAXLEN,
                            corpus_lines=None, on_memory=True)

print("Loading Test Dataset", test_dataset)
test_dataset = bert_pytorch.dataset.BERTDataset(test_dataset, vocab, seq_len=MAXLEN, on_memory=True) \
    if test_dataset is not None else None

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=8, num_workers=4)

test_data_loader = DataLoader(test_dataset, batch_size=64, num_workers=4) \
    if test_dataset is not None else None

print("Building BERT model")
bert = bert_pytorch.BERT(len(vocab), hidden=128, n_layers=3, attn_heads=3, dropout=0.0)

print("Creating BERT Trainer")
trainer = trainer.BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                        lr=1e-5, betas=(0.9, 0.999), weight_decay=0,
                        with_cuda=True, cuda_devices=CUDA_DEVICE, log_freq=10)


print("Training Start")
for epoch in range(20):
    trainer.train(epoch)
    trainer.save(epoch, output_path)
    if test_data_loader is not None:
        trainer.test(epoch)     