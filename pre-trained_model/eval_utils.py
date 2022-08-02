from torch.autograd import Variable
import torch
import re
import numpy

from torch import nn
import torch.nn.functional as F

from config import *
import vocab


# this function is how I parse and pre-pocess instructions for palmtree. It is very simple and based on regular expressions. 
# If I use IDA pro or angr instead of Binary Ninja, I would have come up with a better solution.

def parse_instruction(ins, symbol_map, string_map):
    # arguments:
    # ins: string e.g. "mov, eax, [rax+0x1]"
    # symbol_map: a dict that contains symbols the key is the address and the value is the symbol 
    # string_map : same as symbol_map in Binary Ninja, constant strings will be included into string_map 
    #              and the other meaningful strings like function names will be included into the symbol_map
    #              I think you do not have to separate them. This is just one of the possible nomailization stretagies.
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    token_lst = []
    if len(parts) > 1:
        operand = parts[1:]
    token_lst.append(parts[0])
    for i in range(len(operand)):
        # print(operand)
        symbols = re.split('([0-9A-Za-z]+)', operand[i])
        symbols = [s.strip() for s in symbols if s]
        processed = []
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) > 6 and len(symbols[j]) < 15: 
                # I make a very dumb rule here to treat number larger than 6 but smaller than 15 digits as addresses, 
                # the others are constant numbers and will not be normalized.
                if int(symbols[j], 16) in symbol_map:
                    processed.append("symbol")
                elif int(symbols[j], 16) in string_map:
                    processed.append("string")
                else:
                    processed.append("address")
            else:
                processed.append(symbols[j])
            processed = [p for p in processed if p]

        token_lst.extend(processed) 

    # the output will be like "mov eax [ rax + 0x1 ]"
    return ' '.join(token_lst)



class UsableTransformer:
    def __init__(self, model_path, vocab_path):
        print("Loading Vocab", vocab_path)
        self.vocab = vocab.WordVocab.load_vocab(vocab_path)
        print("Vocab Size: ", len(self.vocab))
        self.model = torch.load(model_path)
        self.model.eval()
        if USE_CUDA:
            self.model.cuda(CUDA_DEVICE)


    def encode(self, text, output_option='lst'):

        segment_label = []
        sequence = []
        for t in text:
            l = (len(t.split(' '))+2) * [1]
            s = self.vocab.to_seq(t)
            # print(t, s)
            s = [3] + s + [2]
            if len(l) > 20:
                segment_label.append(l[:20])
            else:
                segment_label.append(l + [0]*(20-len(l)))
            if len(s) > 20:
                 sequence.append(s[:20])
            else:
                sequence.append(s + [0]*(20-len(s)))
         
        segment_label = torch.LongTensor(segment_label)
        sequence = torch.LongTensor(sequence)

        if USE_CUDA:
            sequence = sequence.cuda(CUDA_DEVICE)
            segment_label = segment_label.cuda(CUDA_DEVICE)

        encoded = self.model.forward(sequence, segment_label)
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