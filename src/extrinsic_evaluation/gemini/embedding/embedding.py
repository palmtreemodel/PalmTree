from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# to use tfdbg
# wrap session object with debugger wrapper
from tensorflow.python import debug as tf_debug
from random import shuffle
from scipy.linalg import block_diag

import tensorflow as tf
import numpy as np
import os
import operator
import time
import pickle as p
import scipy


# local library
from siamese_emb import Siamese

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('vector_size', 128, "Vector size of acfg")
flags.DEFINE_integer('emb_size', 64, "Embedding size for acfg")
flags.DEFINE_float('learning_rate', 0.001, "Learning Rate for Optimizer")
flags.DEFINE_string('data_file', 'train.pickle', "Stores the train sample after preprocessing")
flags.DEFINE_string('test_file', 'test.pickle', "Stores the test sample after preprocessing")
flags.DEFINE_integer('T', 5, "Number of time to be interated while embedding generation")
flags.DEFINE_string('emb_type', 'trans', "Embedding type")
PROJ_DIR = os.path.dirname(os.path.realpath(__file__))


class Embedding:
	def __init__(self):
		self.emb_model_loc = PROJ_DIR + "/model/"
		# self.siamese = Siamese()
		self._init_tensorflow()
		self.g_embed_funcs = [self.siamese.get_embedding()]
		self.g_test_similarity = self.test_similarity_internal()

	def _init_tensorflow(self):
		self.siamese = Siamese()
		global_step = tf.Variable(0, name="global_step", trainable=False)
		print("siamese model  object initialized")

		init_op = tf.global_variables_initializer()


		# set cpu utilization
		#config = tf.ConfigProto(device_count = {'CPU': 1})
		#self.tf_sess = tf.Session(config=config)
		# set gpu utilization %
		#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.001)

		#self.tf_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

		# original
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.tf_sess = tf.Session(config=config)

		# to be used later
		self.tf_saver = tf.train.Saver()
		# can use other optimizers
		optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
		# optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
		train_op = optimizer.minimize(self.siamese.loss)
		print("defined training operations")

		print("initializing global variables")
		self.tf_sess.run(init_op)
		self.tf_saver.restore(self.tf_sess, self.emb_model_loc + "model.ckpt")

	def _close_tensorflow(self):
		self.tf_sess.close()
		del self.siamese

		#print(self.W_1)
	def train_model_all(self):
		print("Retrain the model")

	def embed_single_function(self, acfg_mat, acfg_nbr):

		r4 = self.tf_sess.run([self.siamese.get_embedding()],
							  feed_dict={self.siamese.x: acfg_mat, self.siamese.n: acfg_nbr})

		return r4[0]

	def embed_function_by_name(self, funcname, idafile):
		print("Embedding a single binary function " + funcname)
		# load '.ida' file
		with open(idafile, 'rb') as f:
			acfgs = p.load(f)
		for acfg in acfgs.raw_graph_list:
			fvec_list = []
			func_name = acfg.funcname
			if func_name != funcname:
				continue
			# This loop is to delete first two elements of feature vectors
			# because they are list and we need numeric valure for our matrix
			# if there is method to convert those list to values this loop can be commented out

			for fv in acfg.fv_list:
				# deleting first 2 element of each feature vector
				del fv[:2]
				fvec_list.append(fv)

			# converting to matrix form
			acfg_mat = np.array(fvec_list)

			# setting up neighbor matrix from edge list
			num_nodes = len(fvec_list)
			acfg_nbr = np.zeros((num_nodes, num_nodes))

			for edge in acfg.edge_list:
				acfg_nbr.itemset((edge[0], edge[1]), 1)
				acfg_nbr.itemset((edge[1], edge[0]), 1)

			# embeding function
			embed = self.embed_single_function(acfg_mat, acfg_nbr)
			return tuple([func_name, embed])

		return None

	def get_some_embedding(self, it, cnt=35):
		mul_mat = []
		acfg_mat = []
		acfg_nbr_mat = []
		func_name_list = []
		acfg_length_list = []

		while len(func_name_list) < cnt:
			try:
				acfg = next(it)
			except StopIteration:
				break
			# if len(acfg.fv_list) < 5:
			# 	continue
			fvec_list = []
			func_name = acfg.funcname
			fsize_list = []
			# test print function name
			# print(func_name)

			# This loop is to delete first two elements of feature vectors
			# because they are list and we need numeric valure for our matrix
			# if there is method to convert those list to values this loop can be commented out

			for fv in acfg.fv_list:
				# deleting first 2 element of each feature vector
				fvec_list.append(fv)
				fsize_list.append(len(fv))
			mul_mat.append(np.ones(len(acfg.fv_list)))

			# test fvec shape
			# converting to matrix form

			# initialize acfg_mat
			acfg_mat_tmp = np.concatenate(fvec_list)
			acfg_mat.append(acfg_mat_tmp)
			acfg_length_list.append(fsize_list)
			# matrix input acfg_mat & acfg_nbr
			num_nodes = len(fvec_list)
			acfg_nbr = np.zeros((num_nodes, num_nodes))

			for edge in acfg.edge_list:
				acfg_nbr.itemset((edge[0], edge[1]), 1)
				acfg_nbr.itemset((edge[1], edge[0]), 1)

			acfg_nbr_mat.append(acfg_nbr)
			func_name_list.append(func_name)
		if len(mul_mat) != 0:
			# acfg_mat = np.vstack(acfg_mat)
			acfg_mat = np.concatenate(acfg_mat)
			acfg_nbr_mat = block_diag(*acfg_nbr_mat)
			mul_mat = block_diag(*mul_mat)
		return acfg_mat, acfg_nbr_mat, acfg_length_list,  mul_mat, func_name_list

	def embed_a_binary(self, idafile, target_name=None):
		# counter for test first 100 function
		with open(idafile, 'rb') as f:
			acfgs = p.load(f)

		time_embdding_start = time.time()

		# print shape of acfg_mat & acfg_nbr
		it = iter(acfgs.raw_graph_list)
		retval = []
		func_names = []

		while True:
			acfg_mat, acfg_nbr_mat, acfg_length_list, mul_mat, func_name_list = self.get_some_embedding(it)
			if len(mul_mat) == 0:
				break
			idx = 0
			idy = 0
			merged_acfg_mat = np.ndarray((acfg_nbr_mat.shape[0], FLAGS.vector_size))
			if FLAGS.emb_type != "org":
				for length in acfg_length_list:
					for l in length:
						ins = np.expand_dims(acfg_mat[idx: idx+l], axis=0)
						merged_acfg_mat[idy,:] = np.squeeze(self.tf_sess.run([self.siamese.bb_emb], feed_dict={self.siamese.ins: ins}), axis=0)
						idy += 1
						idx += l
			# print(merged_acfg_mat.shape, acfg_nbr_mat.shape)
				emb = self.tf_sess.run(self.g_embed_funcs, feed_dict={self.siamese.x: np.concatenate([merged_acfg_mat, np.transpose(mul_mat)], 1),
													self.siamese.n: acfg_nbr_mat})
			else:
				emb = self.tf_sess.run(self.g_embed_funcs, feed_dict={self.siamese.x: np.concatenate([acfg_mat, np.transpose(mul_mat)], 1),
												self.siamese.n: acfg_nbr_mat})

			retval.extend(emb[0])
			func_names.extend(func_name_list)

		time_embdding_end = time.time()
		# print("embedding duration: ", time_embdding_end-time_embdding_start)

		if target_name is not None:
			return retval[func_names.index(target_name)]

		# return embedding_list

		return func_names, retval

	def embed_multiple_function(self, acfg_mat, acfg_nbr):
		r5 = self.tf_sess.run(self.g_embed_funcs,
							  feed_dict={self.siamese.x: acfg_mat, self.siamese.n: acfg_nbr})
		return r5[0]

	def test_similarity_internal(self):
		self.funca = tf.placeholder(tf.float32, (None, None))
		self.funcb = tf.placeholder(tf.float32, (None, None))
		mul = tf.matmul(self.funca, self.funcb, transpose_b=True)
		na = tf.norm(self.funca, axis=1, keepdims=True)
		nb = tf.norm(self.funcb, axis=1, keepdims=True)
		return mul / tf.matmul(na, nb, transpose_b=True)

	def test_similarity(self, funca, funcb):
		# funca: embeddings of list a
		# funcb : embeddings of list b
		# ret: predicted value
		return self.tf_sess.run(self.g_test_similarity, feed_dict={self.funca: funca, self.funcb: funcb})

	def gen_pca(self, emb, dims_rescaled_data=2):
		emb = np.array(emb).T
		emb -= emb.mean(axis=0)
		r = np.cov(emb, rowvar=False)
		evals, evecs = scipy.linalg.eigh(r)
		idx = np.argsort(evals)[::-1]
		evecs = evecs[:, idx]
		evecs = evecs[:, :dims_rescaled_data]
		return np.dot(emb, evecs).T.reshape(dims_rescaled_data*64)


'''
	def train_siamese(self):
		# Training
		# ==================================================
		print("starting graph def")
		with tf.Graph().as_default():
			#init class
			siamese = Siamese()
			global_step = tf.Variable(0, name="global_step", trainable=False)
			print("siamese model  object initialized")

			init_op = tf.global_variables_initializer()

			print("started session")
			sess = tf.Session()
			#to be used later
			saver = tf.train.Saver()
			with sess.as_default() as sess:
				#can use other optimizers
				optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
				#optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
				train_op = optimizer.minimize(siamese.loss)
				print("defined training operations")

				print("initializing global variables")
				sess.run(init_op)
				saver.restore(sess, "/tmp/model.ckpt")

				#Implement AUC
				#this part can be parallelized for better embedding generation speed
				print("generating embedding...")
				emb_list = []
				#generating embedding for all acfg in the test sample
				for i, item in enumerate(pair_sample):
					if i%100 == 0:
						print("calucating :", i)
					#print(item)
					r4 = sess.run([siamese.get_embedding()],feed_dict = {siamese.x: item[1], siamese.n: item[2]})
					#print(r4[0])
					#appending generated embedding and name of the function
					emb_list.append((item[0],r4[0]))
					#just for testing small sample
					#if i == 10:
					#	break
'''
