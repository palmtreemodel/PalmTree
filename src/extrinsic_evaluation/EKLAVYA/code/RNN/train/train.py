import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src/skip-thoughts'))
	print(os.getcwd())
except:
	pass
import torch
from torch import nn
from torch.autograd import Variable
import re
import pickle
import random
import numpy as np
from data_loader import DataLoader
from model import UniSkip
from config import *
from datetime import datetime, timedelta

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders import str2enc
from onmt.model_builder import *
from onmt.decoders import str2dec

from onmt.modules import Embeddings, VecEmbedding, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser

from sklearn.metrics import pairwise_distances

data_list = ['data/training/dfg/dfg-seq' + str(i) + '.txt' for i in range(200)]

d = DataLoader(data_list)

mod = UniSkip(model_type='quick-thought', attention=False)

if USE_CUDA:
    mod.cuda(CUDA_DEVICE)

lr = 1e-5
optimizer = torch.optim.Adam(params=mod.parameters(), lr=lr)


loss_trail = []
last_best_loss = None
current_time = datetime.utcnow()

# def fix_key(s):
#             s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
#                        r'\1.layer_norm\2.bias', s)
#             s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
#                        r'\1.layer_norm\2.weight', s)
#             return s

def debug(i, loss):
    global loss_trail
    global last_best_loss
    global current_time

    this_loss = loss.data.item()
    loss_trail.append(this_loss)
    loss_trail = loss_trail[-20:]
    new_current_time = datetime.utcnow()
    time_elapsed = str(new_current_time - current_time)
    current_time = new_current_time
    print("Iteration {}: time = {} last_best_loss = {}, this_loss = {}".format(
              i, time_elapsed, last_best_loss, this_loss))

    # for i in range(3,12):
    #     _, pred_ids = pred[i+1].max(1)
    #     print("current = {}\npred = {}".format(
    #         d.convert_indices_to_sentences(prev[i]),
    #         d.convert_indices_to_sentences(pred_ids)
    #     ))
    #     print("=============================================")
    
    try:
        trail_loss = sum(loss_trail)/len(loss_trail)
        if last_best_loss is None or last_best_loss > trail_loss:
            print("Loss improved from {} to {}".format(last_best_loss, trail_loss))
            
            save_loc = "./saved_models/skip-best".format(lr, VOCAB_SIZE)
            print("saving model at {}".format(save_loc))
            torch.save(mod.state_dict(), save_loc)
            
            last_best_loss = trail_loss

            #save embeddings:
    except Exception as e:
       print("Couldn't save model because {}".format(e))

print("Starting training...")



for i in range(0, 400000):
    positive_samples, positive_context, negative_context = d.fetch_batch_w_neg_sampling(128)
    loss, prev, pred = mod(positive_samples, positive_context, negative_context)
    if i % 2000 == 0:
        debug(i, loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

