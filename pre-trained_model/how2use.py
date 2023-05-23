import os
import re
import sys
from config import *
from torch import nn
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
import torch
import numpy as np
import eval_utils as utils


palmtree = utils.UsableTransformer(
    model_path="./palmtree/transformer.ep19", vocab_path="./palmtree/vocab")

# tokens has to be seperated by spaces.

if 'input.asm' in os.listdir():
    text = open('input.asm').readlines()
    for idx, line in enumerate(text):
        #breakpoint()
        text[idx] = re.sub(r'([\[\]]|\-0x\w+|\+0x\w+|[\-\+](?!0x))', r' \1 ', line) \
            .replace(',','') \
            .strip('\n')
    print(text)
    sys.exit(0)
else:
    text = ["mov rbp rdi",
            "mov ebx 0x1",
            "mov rdx rbx",
            "call memcpy",
            "mov [ rcx + rbx ] 0x0",
            "mov rcx rax",
            "mov [ rax ] 0x2e"]

# it is better to make batches as large as possible.
embeddings = palmtree.encode(text)
print("usable embedding of this basicblock:", embeddings)
print("the shape of output tensor: ", embeddings.shape)
