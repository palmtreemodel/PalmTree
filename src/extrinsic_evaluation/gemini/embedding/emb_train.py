#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Siamese graph embedding implementaition using tensorflow

By:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pkl
import time
import random
import nltk
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.linalg import block_diag
from sklearn import metrics
# from embedding import Embedding
from dataset import BatchGenerator
# local library%
from siamese_emb import Siamese

# to use tfdbg
# wrap session object with debugger wrapper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('vector_size',128, "Vector size of acfg")
flags.DEFINE_integer('emb_size', 64, "Embedding size for acfg")
flags.DEFINE_float('learning_rate', 0.0001, "Learning Rate for Optimizer")
flags.DEFINE_string('data_file', 'train.pickle', "Stores the train sample after preprocessing")
flags.DEFINE_string('test_file', 'test.pickle', "Stores the test sample after preprocessing")
flags.DEFINE_integer('T', 5, "Number of time to be interated while embedding generation")
flags.DEFINE_string('emb_type', 'mlm_only', "Embedding type")

FILTER_SIZE = 2


def get_some_embedding(it, cnt=35):
    acfg_mat = []
    acfg_nbr_mat = []
    acfg_length_list = []
    mul_mat = []
    func_name_list = []

    while len(func_name_list) < cnt:
        try:
            data = next(it)
        # data = it
        except StopIteration:
            break
        func_name = data[0]
        acfg = data[1]
        if len(acfg) < FILTER_SIZE:
            continue
        acfg_nbr = data[2]
        acfg_length = data[3]

        func_name_list.append(func_name)
        acfg_mat.append(acfg)
        acfg_length_list.append(acfg_length)
        acfg_nbr_mat.append(acfg_nbr)
        mul_mat.append(np.ones(len(acfg_nbr)))
    if len(mul_mat) != 0:
        # acfg_mat = np.vstack(acfg_mat)
        acfg_mat = np.concatenate(acfg_mat)
        acfg_nbr_mat = block_diag(*acfg_nbr_mat)
        mul_mat = block_diag(*mul_mat)
    return acfg_mat, acfg_nbr_mat, acfg_length_list,  mul_mat, func_name_list


class Training:
    def __init__(self):
        self.g_test_similarity = self.test_similarity_internal()

    def test_similarity_internal(self):
        self.funca = tf.placeholder(tf.float32, (None, None))
        self.funcb = tf.placeholder(tf.float32, (None, None))

        mul = tf.matmul(self.funca, self.funcb, transpose_b=True)
        na = tf.norm(self.funca, axis=1, keepdims=True)
        nb = tf.norm(self.funcb, axis=1, keepdims=True)
        return mul / tf.matmul(na, nb, transpose_b=True)

    def test_similarity(self, sess, funca, funcb):
        # funca: embeddings of list a
        # funcb : embeddings of list b
        # ret: predicted value
        return sess.run(self.g_test_similarity, feed_dict={self.funca: funca, self.funcb: funcb})


def train_siamese(num_of_iterations):
    # Training part
    print("starting graph def")
    with tf.Graph().as_default():
        # init class
        siamese = Siamese()
        data_gen = BatchGenerator(r'/home/administrator/zixiang/gemini/{}/train/*.ida'.format(FLAGS.emb_type,FLAGS.emb_type), FILTER_SIZE)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        print("siamese model  object initialized")

        

        print("started session")
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))

        saver = tf.train.Saver()
        with sess.as_default() as sess:

            # can use other optimizers
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            
            # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            train_op = optimizer.minimize(siamese.loss)
            init_op = tf.global_variables_initializer()
            print("defined training operations")
            print("initializing global variables")

            sess.run(init_op)
            # Trainning parameters
            TRAIN_ITER = num_of_iterations  # number of iterations in each training
            # model saved path
            SAVEPATH = "./model/model.ckpt"

            ## start of counting training time
            time_train_start = time.time()

            print("model training start:")
            # Temporary loss value
            temp_loss_r1 = 10
            temp_loss_r3 = 10
            for i in range(1, TRAIN_ITER):  ## default 1k, set to 100 for test
                g, g1, g2 = data_gen.get_train_acfg()

                if FLAGS.emb_type == 'manual': #or FLAGS.emb_type == 'cfg_bert' or FLAGS.emb_type == 'mlm_only':
                    bb = g[1]
                    bb1 = g1[1]
                    bb2 = g2[1]
                else:
                    with tf.variable_scope("acfg_embedding") as siam_scope:
                        idx = 0
                        bb = []
                        for length in g[3]:
                            ins = np.expand_dims(g[1][idx: idx+length], axis=0)
                            bb.append(sess.run([siamese.bb_emb], feed_dict={siamese.ins: ins}))
                            idx += length
                        bb = np.reshape(np.array(bb),(-1, FLAGS.vector_size))
                        siam_scope.reuse_variables()

                        idx = 0
                        bb1 = []
                        for length in g1[3]:
                            ins = np.expand_dims(g1[1][idx: idx+length], axis=0)
                            bb1.append(sess.run([siamese.bb_emb], feed_dict={siamese.ins: ins}))
                            idx += length

                        bb1 = np.reshape(np.array(bb1),(-1, FLAGS.vector_size))
                        siam_scope.reuse_variables()

                        idx = 0
                        bb2 = []
                        for length in g2[3]:
                            ins = np.expand_dims(g2[1][idx: idx+length], axis=0)
                            bb2.append(sess.run([siamese.bb_emb], feed_dict={siamese.ins: ins}))
                            idx += length

                        bb2 = np.reshape(np.array(bb2),(-1, FLAGS.vector_size))
                    
                #import pdb; pdb.set_trace()
                #print("x: ", bb.shape)
                #print("n: ", g[2].shape)
                r0, r1 = sess.run([train_op, siamese.loss],
                                  feed_dict={siamese.x1: bb, siamese.x2: bb1, siamese.y: 1,
                                             siamese.n1: g[2], siamese.n2: g1[2]})
                r2, r3 = sess.run([train_op, siamese.loss],
                                  feed_dict={siamese.x1: bb, siamese.x2: bb2, siamese.y: -1,
                                             siamese.n1: g[2], siamese.n2: g2[2]})
                # currently saving for best loss modify to save for best AUC
                if i% 10 == 0:
                    # Save the variables to disk.
                    if r3 < temp_loss_r3:
                        saver.save(sess, SAVEPATH)
                        print("Model saved", i, r1 , r3)
                        temp_loss_r3 = r3
                        continue
                    if r1 < temp_loss_r1:
                        saver.save(sess, SAVEPATH)
                        print("Model saved", i, r1 , r3)
                        temp_loss_r1 = r1
                        continue
                        

            ## Restore variables from disk for least loss.
            ## To be changed for best AUC
            # end of counting training time
            time_train_end = time.time()
            # get total training time
            print("traing duration: ", time_train_end - time_train_start)

            # evalution part
            saver.restore(sess, SAVEPATH)

            print("generating embedding for test samples")
            emb_list = []
            name_list = []
            test_list = []
            data_gen = BatchGenerator(r'/home/administrator/zixiang/gemini/{}/test/*.ida'.format(FLAGS.emb_type,FLAGS.emb_type), FILTER_SIZE)   
            for k, v in data_gen.train_sample.items():
                if len(v) >= 2:
                    rd = random.sample(v, 2)
                    test_list.extend(rd)
            it = iter(test_list)
            emb_func = [siamese.get_embedding()]

            while True:
                acfg_mat, acfg_nbr_mat, acfg_length_list, mul_mat, func_name_list = get_some_embedding(it)
                if len(mul_mat) == 0:
                    break
                idx = 0
                idy = 0
                merged_acfg_mat = np.ndarray((acfg_nbr_mat.shape[0], FLAGS.vector_size))
                if FLAGS.emb_type != "manual" and FLAGS.emb_type != "albert_avg":
                    for length in acfg_length_list:
                        for l in length:
                            ins = np.expand_dims(acfg_mat[idx: idx+l], axis=0)
                            merged_acfg_mat[idy,:] = np.squeeze(sess.run([siamese.bb_emb], feed_dict={siamese.ins: ins}), axis=0)
                            idy += 1
                            idx += l
                # print(merged_acfg_mat.shape, acfg_nbr_mat.shape)
                    emb = sess.run(emb_func, feed_dict={siamese.x: np.concatenate([merged_acfg_mat, np.transpose(mul_mat)], 1),
                                                        siamese.n: acfg_nbr_mat})
                else:
                    emb = sess.run(emb_func, feed_dict={siamese.x: np.concatenate([acfg_mat, np.transpose(mul_mat)], 1),
                                                    siamese.n: acfg_nbr_mat})

                emb_list.extend(emb[0])
                name_list.extend(func_name_list)

            print("evaluating prediction values")
            # AUC_Matrix = []
            # done_pred = []

            training = Training()
            resultMat = training.test_similarity(sess, emb_list, emb_list)

            rank_index_list = []

            to_sort_list = tf.placeholder(tf.float32, (None, None))
            sort_func = tf.contrib.framework.argsort(to_sort_list, direction='DESCENDING')

            time_eval_start = time.time()
            for i in range(0, len(resultMat), 5000):
                ret = sess.run(sort_func, feed_dict={to_sort_list: resultMat[i:i + 5000]})
                rank_index_list.extend(ret)
            time_eval_end = time.time()

            print("sort duration: ", time_eval_end - time_eval_start)
            # for i in range(len(resultMat)):
            #     sample = resultMat[i]
                
            #     if (name_list[i] != name_list[rank_index_list[i][0]] and i != rank_index_list[i][0]) or (name_list[i] != name_list[rank_index_list[i][1]] and i == rank_index_list[i][0]):
            #         sample = np.sort(sample)
            #         print(sample[-50:])
            #         print("target:", name_list[i])
            #         print("candidate: ", [name_list[j] for j in rank_index_list[i][:50]])

            del resultMat

            func_counts = len(rank_index_list)
            print("func_counts: ", func_counts)
            total_tp = []
            total_fp = []
            for func in range(func_counts):
                real_name = name_list[func]
                tp = [0]
                fp = [0]
                for rank, idx in enumerate(rank_index_list[func]):
                    if func == idx:
                        assert name_list[idx] == real_name
                        continue
                    if name_list[idx] == real_name:
                        #print(rank)
                        tp.append(1)
                        fp.append(fp[-1])
                    else:
                        tp.append(max(tp[-1], 0))
                        fp.append(fp[-1] + 1)
                total_tp.append(tp[1:])
                total_fp.append(fp[1:])
            # num_positive = sum(len(v) * len(v) for k, v in data_gen.test_sample.items())
            num_positive = len(test_list)
            num_negative = func_counts * func_counts - num_positive - func_counts
            
            total_tp = np.sum(total_tp, axis=0, dtype=np.float) / func_counts
            total_fp = np.sum(total_fp, axis=0, dtype=np.float) / num_negative
            time_eval_end = time.time()
            print("eval duration: ", time_eval_end - time_eval_start)

        return total_fp, total_tp

    

def plot_eval_siamese(total_fp, total_tp):
    plt.figure(1)
    plt.title('ROC')
    plt.plot(total_fp, total_tp, '-', label='ROC')
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    total_fp, total_tp = train_siamese(40001)
    plot_eval_siamese(total_fp, total_tp)
    with open('./{}_total_fp.txt'.format(FLAGS.emb_type), 'wb') as f:
        pkl.dump(total_fp, f)
    with open('./{}_total_tp.txt'.format(FLAGS.emb_type), 'wb') as f:
        pkl.dump(total_tp, f)
    
    print(metrics.auc(total_fp, total_tp))
