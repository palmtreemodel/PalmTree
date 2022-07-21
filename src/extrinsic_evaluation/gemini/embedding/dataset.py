import glob
import random
from collections import defaultdict

import tensorflow as tf
import numpy as np
import pickle as p
from numpy.random import choice, permutation
from itertools import combinations
import util
import os
import sys
import re
import operator
from functools import reduce


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir) + "/coogleconfig")

from random import shuffle

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from raw_graphs import *

flags = tf.app.flags
FLAGS = flags.FLAGS


# Added to test standalone testing of this filef
# To be commented if testing with NN Code
# flags.DEFINE_string('data_file', 'train.pickle', "Stores the train sample after preprocessing")
# flags.DEFINE_string('test_file', 'test.pickle', "Stores the test sample after preprocessing")

class BatchGenerator():
    # __slots__=('filter_size', 'train_sample', 'test_sample')
    def __init__(self, training_dataset, filter_size=0):
        np.random.seed()
        self.filter_size = filter_size
        self.train_sample = defaultdict(list)
        self.create_sample_space(training_dataset)

    # for testing
    # need to commented out while training
    # g, g1, g2 = self.get_train_acfg()
    # print("g: ", g)
    # print("g1: ", g1)
    # print("g2: ", g2)

    # create sample space from all the available '.ida' files
    def create_sample_space(self, training_dataset):
        for ida_path in glob.iglob(training_dataset):
            # load '.ida' file
            print("train:",ida_path)
            acfgs = p.load(open(ida_path, 'rb'))
            
            filter_cnt = 0
            for acfg in acfgs.raw_graph_list:
                # if len(reduce(operator.add, acfg.fv_list)) < self.filter_size:
                if len(acfg.fv_list) < self.filter_size:
                    filter_cnt += 1
                    continue
                fvec_list = []
                fsize_list = []
                func_name = acfg.funcname

                # This loop is to delete first two elements of feature vectors
                # because they are list and we need numeric valure for our matrix
                # if there is method to convert those list to values this loop can be commented out
                for fv in acfg.fv_list:
                    # deleting first 2 element of each feature vector
                    # del fv[:2]
                    fvec_list.append(fv)
                    if FLAGS.emb_type != 'org': 
                        fsize_list.append(len(fv))
                    else:
                        fsize_list.append(1)


                # converting to matrix form
                if FLAGS.emb_type == "manual": 
                    acfg_mat = np.array(fvec_list)
                else:
                    acfg_mat = np.concatenate(fvec_list)

                # func_data = tuple(acfg_mat.ravel().tolist())


                # if FLAGS.emb_type == 'org' and func_name in func_name_filter and func_data in func_filter:
                #     filter_cnt += 1
                #     continue

                # if func_data in func_filter:
                #     filter_cnt += 1
                #     continue 

                # if FLAGS.emb_type == 'manual':
                #     func_name_filter.add(func_name)

                # func_filter.add(func_data)
                # setting up neighbor matrix from edge list
                num_nodes = len(fsize_list)
                acfg_nbr = np.zeros((num_nodes, num_nodes))
                
                for edge in acfg.edge_list:
                    acfg_nbr.itemset((edge[0], edge[1]), 1)
                    acfg_nbr.itemset((edge[1], edge[0]), 1)

                self.train_sample[func_name].append((func_name, acfg_mat, acfg_nbr, fsize_list))
            print(filter_cnt, len(acfgs.raw_graph_list))



        # # divide the training and testing data
        # test_func_filter = set()
        # test_func_name_filter = set()
        # for test_ida_path in glob.iglob(testing_dataset):
        #     # load '.ida' file
        #     print("test:", test_ida_path)
        #     test_acfgs = p.load(open(test_ida_path, 'rb'))

        #     test_filter_cnt = 0

        #     for acfg in test_acfgs.raw_graph_list:
        #         # if len(reduce(operator.add, acfg.fv_list)) < self.filter_size:
        #         if len(acfg.fv_list) < self.filter_size:
        #             test_filter_cnt += 1
        #             continue
        #         fvec_list = []
        #         fsize_list = []
        #         func_name = acfg.funcname

        #         # This loop is to delete first two elements of feature vectors
        #         # because they are list and we need numeric valure for our matrix
        #         # if there is method to convert those list to values this loop can be commented out
        #         for fv in acfg.fv_list:
        #             # deleting first 2 element of each feature vector
        #             # del fv[:2]
        #             fvec_list.append(fv)
        #             fsize_list.append(len(fv))

        #         # converting to matrix form
        #         if FLAGS.emb_type == "manual":
        #             acfg_mat = np.array(fvec_list)
        #         else:
        #             acfg_mat = np.concatenate(fvec_list)
        #         func_data = tuple(acfg_mat.ravel().tolist())

        #         # if FLAGS.emb_type == 'org' and (func_data in test_func_filter) and (func_name in test_func_name_filter):
        #         #     test_filter_cnt += 1
        #         #     continue
        #         if func_data in test_func_filter:
        #             test_filter_cnt += 1
        #             continue 


        #         if FLAGS.emb_type == 'manual':
        #             test_func_name_filter.add(func_name)  
        #         test_func_filter.add(func_data)
        #         # setting up neighbor matrix from edge list
        #         num_nodes = len(fsize_list)
        #         acfg_nbr = np.zeros((num_nodes, num_nodes))

        #         for edge in acfg.edge_list:
        #             acfg_nbr.itemset((edge[0], edge[1]), 1)
        #             acfg_nbr.itemset((edge[1], edge[0]), 1)

        #         self.test_sample[func_name].append((func_name, acfg_mat, acfg_nbr, fsize_list))
        #     print(test_filter_cnt, len(test_acfgs.raw_graph_list))


    # get train acfg
    def get_train_acfg(self):
        return self.get_acfg_pairs(self.train_sample)

    # get test acfg
    def get_test_acfg(self):
        return self.get_acfg_pairs(self.test_sample)

    # get randomly selected acgf pair sampled from sample list
    def get_acfg_pairs(self, sample): 
        while True: 
            k1, k2 = np.random.choice(list(sample.keys()), 2, False)
            if len(sample[k1]) > 1:
                break
        idx1, idx2 = np.random.choice(len(sample[k1]), 2, False)
        g, g1 = sample[k1][idx1], sample[k1][idx2]
        g2 = random.choice(sample[k2])
        return g, g1, g2


    def split_function_name(self, s):
        s = re.sub('\d', ' ', s)
        s = re.sub('_', ' ', s)
        tokens = s.split(' ')
        tokens_f = []
        for t in tokens:
            res_list = re.findall('[A-Z][^A-Z]+', t)
            if len(res_list) > 0:
                tokens_f.extend(res_list)
            if len(res_list) == 0 and len(t) != 0:
                tokens_f.append(t)
        return tokens_f

    # Divide sample space into training and testing sample
    # def divide_sample_space(self):
    #     sample_size = sum([len(v) for v in self.sample.values()])
    #     train_size = int(sample_size * .5)
    #     keys = list(self.sample.keys())
    #     shuffle(keys)
    #     train_sample = defaultdict(list)
    #     test_sample = defaultdict(list)
    #     it = iter(keys)
    #     total_len = 0
    #     while True:
    #         k = next(it)
    #         train_sample[k] = self.sample[k]
    #         total_len += len(self.sample[k])
    #         if total_len >= train_size:
    #             break
    #     for k in it:
    #         test_sample[k] = self.sample[k]

    #     return train_sample, test_sample


if __name__ == '__main__':
    sample_gen = BatchGenerator()
