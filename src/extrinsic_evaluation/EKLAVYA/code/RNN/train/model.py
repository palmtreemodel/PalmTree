"""
This file implements the Skip-Thought architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import *
import math
import numpy as np

class Encoder(nn.Module):
    thought_size = 128
    word_size = 256

    @staticmethod
    def reverse_variable(var):
        idx = [i for i in range(var.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx))

        if USE_CUDA:
            idx = idx.cuda(CUDA_DEVICE)

        inverted_var = var.index_select(0, idx)
        return inverted_var

    def __init__(self):
        super(Encoder, self).__init__()
        # self.rnn = nn.LSTM(self.word_size, self.thought_size)
        self.rnn = nn.GRU(self.word_size, self.thought_size, bidirectional=False)

    def forward(self, embeddings):
        # sentences = (batch_size, maxlen), with padding on the right.

        # sentences = sentences.transpose(0, 1)  # (maxlen, batch_size)

        # word_embeddings = torch.tanh(self.word2embd(sentences))  # (maxlen, batch_size, word_size)
        output, thoughts = self.rnn(embeddings)
        # _, thoughts = self.rnn(embeddings)

        return output, thoughts

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class Decoder(nn.Module):

    word_size = Encoder.word_size

    def __init__(self, hidden_size, attention):
        super(Decoder,self).__init__()
        # self.rnn = nn.GRU(input_size= Encoder.word_size+Encoder.thought_size, hidden_size=Encoder.thought_size, bidirectional=False)
        # self.worder = nn.Linear(Encoder.thought_size, VOCAB_SIZE)
        self.attention = attention
        if attention:
            self.rnn = nn.GRU(input_size= Encoder.word_size+hidden_size, hidden_size=hidden_size, bidirectional=False)
            self.worder = nn.Linear(hidden_size*2, VOCAB_SIZE)
            self.attn = Attn(hidden_size)
        else:
            self.rnn = nn.GRU(input_size= Encoder.word_size, hidden_size=hidden_size, bidirectional=False)
            self.worder = nn.Linear(hidden_size, VOCAB_SIZE)            

    def forward(self, thoughts, word_embedded, encoder_outputs, decoder_context):
        word_embedded =  word_embedded.view(1, encoder_outputs.size(1), Encoder.word_size)

        # attn_weights = self.attn(thoughts, encoder_outputs)
        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        # context = context.transpose(0, 1)  # (1,B,V)
        # rnn_input = torch.cat((word_embedded, context), 2)
        # output, hidden = self.rnn(rnn_input, thoughts)
        # word = F.log_softmax(self.worder(output), dim=2)
        # word = word.transpose(0, 1).contiguous()
        if self.attention:
            rnn_input = torch.cat((word_embedded, decoder_context), 2)
            output, hidden = self.rnn(rnn_input, thoughts)
            attn_weights = self.attn(output.squeeze(0), encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
            context = context.transpose(0, 1) # (1,B,V)
            word = F.log_softmax(self.worder(torch.cat([output,context],dim=2)), dim=2)
        else:
            output, hidden = self.rnn(word_embedded, thoughts)
            word = F.log_softmax(self.worder(output), dim=2)
            context = None
        
        
        word = word.transpose(0, 1).contiguous()
        
        return word, hidden, context


class UniSkip(nn.Module):

    def __init__(self, model_type='skip-inst', attention=False):
        super(UniSkip, self).__init__()
        self.model_type = model_type
        if self.model_type == 'skip-inst':
            self.word2embd = nn.Embedding(VOCAB_SIZE, Encoder.word_size)
            self.encoder = Encoder()
            self.prev_decoder = Decoder(Encoder.thought_size, attention=attention)
            self.next_decoder = Decoder(Encoder.thought_size, attention=attention)
        elif self.model_type == 'cbo-inst':
            self.word2embd = nn.Embedding(VOCAB_SIZE, Encoder.word_size)
            self.encoder = Encoder()
            self.decoder = Decoder(2*Encoder.thought_size, attention=attention)
        else:
            self.word2embd = nn.Embedding(VOCAB_SIZE, Encoder.word_size)
            self.encoder = Encoder() 
            self.context_encoder = Encoder() 
            self.outputembd = nn.Embedding(VOCAB_SIZE, Encoder.word_size)          

    def forward(self, positive_samples, positive_context, negative_context):
        sentences = positive_samples
        if self.model_type == 'skip-inst':
            sentences = sentences.transpose(0, 1)  # (maxlen, batch_size)
            word_embeddings = torch.tanh(self.word2embd(sentences))

            output, thoughts = self.encoder(word_embeddings[:,1:-1,:])

            prev_context = Variable(torch.zeros([1,sentences.size(1)-2,Encoder.word_size]))
            next_context = Variable(torch.zeros([1,sentences.size(1)-2,Encoder.word_size]))

            prev_decoder_context = Variable(torch.zeros([1,sentences.size(1)-2,Encoder.thought_size]))
            next_decoder_context = Variable(torch.zeros([1,sentences.size(1)-2,Encoder.thought_size]))

            prev_hidden = thoughts
            next_hidden = thoughts
            if USE_CUDA:
                prev_context = prev_context.cuda(CUDA_DEVICE)
                next_context = next_context.cuda(CUDA_DEVICE)
                prev_decoder_context = prev_decoder_context.cuda(CUDA_DEVICE)
                next_decoder_context = next_decoder_context.cuda(CUDA_DEVICE)
            prev_word = []
            next_word = []
            for i in range(MAXLEN):
                prev_context, prev_hidden, prev_decoder_context = self.prev_decoder(prev_hidden, prev_context, output, prev_decoder_context)  # both = (batch-1, maxlen, VOCAB_SIZE)
                next_context, next_hidden, next_decoder_context = self.next_decoder(next_hidden, next_context, output, next_decoder_context)
                
                prev_word.append(prev_context)
                next_word.append(next_context)

                prev_context = torch.tanh(self.word2embd(prev_context.max(2)[1]))
                next_context = torch.tanh(self.word2embd(prev_context.max(2)[1]))

            # print(prev_word.size())
            prev_word = torch.cat(prev_word, dim=1)
            next_word = torch.cat(next_word, dim=1)
            # print(prev_word.size())
            prev_loss = F.cross_entropy(prev_word.view(-1, VOCAB_SIZE), sentences.transpose(0, 1)[:-2, :].view(-1))
            next_loss = F.cross_entropy(next_word.view(-1, VOCAB_SIZE), sentences.transpose(0, 1)[2:, :].view(-1))
            loss = prev_loss + next_loss

            return loss, sentences.transpose(0, 1), prev_word

        elif self.model_type == 'cbo-inst':
            sentences = sentences.transpose(0, 1)  # (maxlen, batch_size)
            word_embeddings = torch.tanh(self.word2embd(sentences))

            prev_output, prev_thoughts = self.encoder(word_embeddings[:,:-2,:])
            next_output, next_thoughts = self.encoder(word_embeddings[:,2:,:])
            
            hidden = torch.cat([prev_thoughts, next_thoughts], dim=2)
            encoder_outputs = torch.cat([prev_output, next_output],dim=2)

            context = Variable(torch.zeros([1,sentences.size(1)-2,Encoder.word_size]))
            decoder_context = Variable(torch.zeros([1,sentences.size(1)-2,Encoder.thought_size*2]))

            if USE_CUDA:
                context = context.cuda(CUDA_DEVICE)
                decoder_context = decoder_context.cuda(CUDA_DEVICE)

            word = []
            for i in range(MAXLEN): 
                if i == 0:
                    embd = Variable(torch.zeros([1,sentences.size(1)-2,Encoder.word_size]))
                    if USE_CUDA:
                        embd = embd.cuda(CUDA_DEVICE)
                else:
                    embd = torch.tanh(self.word2embd(sentences[i-1,1:-1]))
                # print(embd.size())          
                # context, hidden, decoder_context = self.decoder(hidden, context, encoder_outputs, decoder_context)
                context, _, decoder_context = self.decoder(hidden, embd, encoder_outputs, decoder_context)
                word.append(context)
                # context = torch.tanh(self.word2embd(context.max(2)[1]))

            word = torch.cat(word, dim=1)
            # print(word.size())
            loss = F.cross_entropy(word.view(-1, VOCAB_SIZE), sentences.transpose(0,1)[1:-1, :].view(-1))
            return loss, sentences.transpose(0,1), word
        
        elif self.model_type == 'quick-thought':
           
            # sentences = sentences.transpose(0, 1)  # (maxlen, batch_size)
            # samples = samples.transpose(0,1)

            # batch_size = sentences.size(1)-1

            # word_embeddings = torch.tanh(self.word2embd(sentences))
            # sample_embeddings = torch.tanh(self.word2embd(samples))

            # _, thoughts = self.encoder(word_embeddings)
            # thoughts = thoughts.squeeze()


            # _, sample_thoughts = self.encoder(sample_embeddings)
            # sample_thoughts = sample_thoughts.squeeze()


            # positive_samples = torch.sum(torch.mul(thoughts[:-1,:], thoughts[1:,:]),dim=1)
            # negative_samples = torch.sum(torch.mul(thoughts[:-1,:], sample_thoughts[1:,:]), dim=1)


            # positive_target = torch.ones(batch_size, device="cuda:0")
            # negative_target = torch.zeros(batch_size, device="cuda:0")

            # loss_sunc = nn.BCEWithLogitsLoss(reduction='mean')
            # pos_loss = loss_sunc(positive_samples, positive_target)
            # neg_los = loss_sunc(negative_samples, negative_target)
            # loss = 0.7*pos_loss + 0.3*neg_los

            positive_samples = positive_samples.transpose(0, 1)
            positive_context = positive_context.transpose(0, 1)
            negative_context = negative_context.transpose(0, 1)

            batch_size = negative_context.size(1)

            positive_emb = torch.tanh(self.word2embd(positive_samples))
            positive_ctxt_emb = torch.tanh(self.outputembd(positive_context))
            negative_ctxt_emb = torch.tanh(self.outputembd(negative_context))

            _, pos_thought = self.encoder(positive_emb)
            pos_thought = pos_thought.squeeze()

            _, pos_ctxt_thought = self.context_encoder(positive_ctxt_emb)
            pos_ctxt_thought = pos_ctxt_thought.squeeze()

            _, neg_ctxt_thought = self.context_encoder(negative_ctxt_emb)
            neg_ctxt_thought = neg_ctxt_thought.squeeze()

            positive_samples = torch.sum(torch.mul(pos_thought, pos_ctxt_thought),dim=1)
            negative_samples = torch.sum(torch.mul(pos_thought, neg_ctxt_thought),dim=1)

            positive_target = torch.ones(batch_size, device="cuda:0")
            negative_target = torch.zeros(batch_size, device="cuda:0")
            loss_sunc = nn.BCEWithLogitsLoss(reduction='mean')
            pos_loss = loss_sunc(positive_samples, positive_target)
            neg_los = loss_sunc(negative_samples, negative_target)
            loss = pos_loss + neg_los


            return loss, None, None
 


            # scores = torch.matmul(thoughts[:-1,:], torch.t(thoughts[1:,:]))
            # scores[range(len(scores)), range(len(scores))] = torch.zeros(batch_size, device='cuda:0')
            # targets_np = np.zeros((batch_size, batch_size))
            # ctxt_sent_pos = [-1,1]
            # for ctxt_pos in ctxt_sent_pos:
            #     targets_np += np.eye(batch_size, k=ctxt_pos)
            # targets_np_sum = np.sum(targets_np, axis=1, keepdims=True)
            # targets_np = targets_np/targets_np_sum
            # targets = torch.tensor(targets_np, dtype=torch.float32, requires_grad=True, device='cuda:0')
            # loss_sunc = nn.BCEWithLogitsLoss(reduce=True, reduction='mean')










