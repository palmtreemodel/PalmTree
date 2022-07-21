import argparse
import functools
import inspect
import os
import sys
import pickle

import dataset
import dataset_caller
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


def placeholder_inputs(class_num, max_length= 500, embedding_dim= 256):
    data_placeholder = tf.placeholder(tf.float32, [None, max_length, embedding_dim])
    label_placeholder = tf.placeholder(tf.float32, [None, class_num])
    length_placeholder = tf.placeholder(tf.int32, [None,])
    keep_prob_placeholder = tf.placeholder(tf.float32) # dropout (keep probability)
    return data_placeholder, label_placeholder, length_placeholder, keep_prob_placeholder


def fill_feed_dict(data_set, batch_size, keep_prob, data_pl, label_pl, length_pl, keep_prob_pl):
    data_batch = data_set.get_batch(batch_size=batch_size)
    feed_dict = {
        data_pl: data_batch['data'],
        label_pl: data_batch['label'],
        length_pl: data_batch['length'],
        keep_prob_pl: keep_prob
    }
    return feed_dict

def fill_test_dict(data_set, batch_size, data_tag, keep_prob, data_pl, label_pl, length_pl, keep_prob_pl):
    data_batch = data_set.get_test_batch(batch_size=batch_size)

    feed_dict = {
        data_pl: data_batch['data'],
        label_pl: data_batch['label'],
        length_pl: data_batch['length'],
        keep_prob_pl: keep_prob
    }
    return feed_dict, data_batch['func_name']


class Model(object):
    def __init__(self, session, my_data, config_info, data_pl, label_pl, length_pl, keep_prob_pl):
        self.session = session
        self.datasets = my_data
        self.emb_dim = int(config_info['embed_dim'])
        self.dropout = float(config_info['dropout'])
        self.num_layers = int(config_info['num_layers'])
        self.num_classes = int(config_info['num_classes'])
        self.max_to_save = int(config_info['max_to_save'])
        self.output_dir = config_info['log_path']
        self.batch_size = int(config_info['batch_size'])
        self.summary_frequency = int(config_info['summary_frequency'])
        self.embd_type = config_info['embedding_type']
        self._data = data_pl
        self._label = label_pl
        self._length = length_pl
        self._keep_prob = keep_prob_pl

        self.run_count = 0

        self.build_graph()

    @lazy_property
    def probability(self):
        def lstm_cell():
            if 'reuse' in inspect.getargspec(tf.contrib.rnn.GRUCell.__init__).args:
                return tf.contrib.rnn.GRUCell(self.emb_dim, reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.GRUCell(self.emb_dim)

        attn_cell = lstm_cell
        if self.dropout < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=self._keep_prob)
        single_cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.num_layers)], state_is_tuple=True)

        if self.embd_type == '1hot':
            embedding = layers.Embedding(5001, 128)
            self.emb_ins = keras.activations.tanh(embedding(self._data))
            self.emb_ins = tf.squeeze(self.emb_ins, axis=2)
            output, state = tf.nn.dynamic_rnn(single_cell, self.emb_ins, dtype=tf.float32,
                                            sequence_length=self._length)
        else:
            output, state = tf.nn.dynamic_rnn(single_cell, self._data, dtype=tf.float32,
                                        sequence_length=self._length)
        weight = tf.Variable(tf.truncated_normal([self.emb_dim, self.num_classes], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[self.num_classes]))

        self.output = output
        probability = tf.matmul(self.last_relevant(output, self._length), weight) + bias
        return probability

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_len = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

    @lazy_property
    def cost_list(self):
        prediction = self.probability
        target = self._label
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=target)
        return cross_entropy

    @lazy_property
    def cost(self):
        cross_entropy = tf.reduce_mean(self.cost_list)
        tf.summary.scalar('cross_entropy', cross_entropy)
        return cross_entropy

    @lazy_property
    def optimize(self):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer().minimize(self.cost, global_step)

        return train_op

    @lazy_property
    def calc_accuracy(self):
        true_probability = tf.nn.softmax(self.probability)
        correct_pred = tf.equal(tf.argmax(true_probability, 1), tf.argmax(self._label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('acc', accuracy)
        return accuracy

    @lazy_property
    def pred_label(self):
        true_probability = tf.nn.softmax(self.probability)
        pred_output = tf.argmax(true_probability, 1)
        label_output = tf.argmax(self._label, 1)
        output_result = {
            'pred': pred_output,
            'label': label_output
        }
        return output_result

    def build_graph(self):
        self.optimize
        self.calc_accuracy
        self.pred_label

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.output_dir + '/train', self.session.graph)
        self.test_writer = tf.summary.FileWriter(self.output_dir + '/test')

        self.saver = tf.train.Saver(tf.trainable_variables(),
                                    max_to_keep=self.max_to_save)

        tf.global_variables_initializer().run()

    def train(self):
        feed_dict = fill_feed_dict(self.datasets, self.batch_size, self.dropout,
                       self._data, self._label, self._length, self._keep_prob)
        if self.run_count % self.summary_frequency == 0:
            cost, acc, summary, _ = self.session.run(
                [self.cost, self.calc_accuracy, self.merged, self.optimize],
                feed_dict = feed_dict
            )
            self.train_writer.add_summary(summary, self.run_count)
            print('[Batch %d][Epoch %d] cost: %.3f; accuracy: %.3f' % (self.run_count,
                                                                              self.datasets._complete_epochs,
                                                                       cost,
                                                                       acc))
        else:
            self.session.run(self.optimize, feed_dict = feed_dict)

        self.run_count += 1

    def test(self):
        total_result = {
            'cost': [],
            'pred': [],
            'func_name': [],
            'acc':[]
        }
        while self.datasets.test_tag:
            feed_dict, func_name_list = fill_test_dict(self.datasets, self.batch_size, 'test', 1.0,
                                                       self._data, self._label, self._length, self._keep_prob)
            cost_result, pred_result, acc = self.session.run(
                [self.cost_list, self.pred_label, self.calc_accuracy],
                feed_dict = feed_dict
            )
            print(acc)
            total_result['cost'].append(cost_result)
            total_result['pred'].append(pred_result)
            total_result['func_name'].append(func_name_list)
            total_result['acc'].append(acc)

        return total_result




def training(config_info):
    data_folder = config_info['data_folder']
    func_path = config_info['func_path']
    embed_path = config_info['embed_path']
    tag = config_info['tag']
    data_tag = config_info['data_tag']
    process_num = int(config_info['process_num'])
    embed_dim = int(config_info['embed_dim'])
    max_length = int(config_info['max_length'])
    num_classes = int(config_info['num_classes'])
    epoch_num = int(config_info['epoch_num'])
    save_batch_num = int(config_info['save_batchs'])
    output_dir = config_info['output_dir']
    embd_type = config_info['embedding_type']

    '''create model & log folder'''
    if os.path.exists(output_dir):
        pass
    else:
        os.mkdir(output_dir)
    model_basedir = os.path.join(output_dir, 'model')
    if os.path.exists(model_basedir):
        pass
    else:
        os.mkdir(model_basedir)
    log_basedir = os.path.join(output_dir, 'log')
    if tf.gfile.Exists(log_basedir):
        tf.gfile.DeleteRecursively(log_basedir)
    tf.gfile.MakeDirs(log_basedir)
    config_info['log_path'] = log_basedir
    print('Created all folders!')

    '''load dataset'''
    if data_tag == 'callee':

        my_data = dataset.Dataset(data_folder, func_path, embed_path, process_num, embed_dim, max_length, num_classes, tag, embd_type)
    else: #caller
        my_data = dataset_caller.Dataset(data_folder, func_path, embed_path, process_num, embed_dim, max_length, num_classes, tag)
    
    print('Created the dataset!')

    with tf.Graph().as_default(), tf.Session() as session:
        # generate placeholder
        data_pl, label_pl, length_pl, keep_prob_pl = placeholder_inputs(num_classes, max_length, embed_dim)

        # generate model
        model = Model(session, my_data, config_info, data_pl, label_pl, length_pl, keep_prob_pl)
        print('Created the model!')

        while my_data._complete_epochs < epoch_num:
            model.train()
            if model.run_count % save_batch_num == 0:
                model.saver.save(session, os.path.join(model_basedir, 'model'), global_step = model.run_count)
                print('Saved the model ... %d' % model.run_count)
            else:
                pass
        model.train_writer.close()
        model.test_writer.close()


def get_model_id_list(folder_path):
    file_list = os.listdir(folder_path)
    model_id_set = set()
    for file_name in file_list:
        if file_name[:6] == 'model-':
            model_id_set.add(int(file_name.split('.')[0].split('-')[-1]))
        else:
            pass
    model_id_list = sorted(list(model_id_set))
    return model_id_list





def testing(config_info):
    data_folder = config_info['data_folder']
    func_path = config_info['func_path']
    embed_path = config_info['embed_path']
    tag = config_info['tag']
    data_tag = config_info['data_tag']
    process_num = int(config_info['process_num'])
    embed_dim = int(config_info['embed_dim'])
    max_length = int(config_info['max_length'])
    num_classes = int(config_info['num_classes'])
    model_dir = config_info['model_dir']
    output_dir = config_info['output_dir']
    embd_type = config_info['embedding_type']


    '''create model & log folder'''
    log_basedir = os.path.join(output_dir, 'log')
    if tf.gfile.Exists(log_basedir):
        # tf.gfile.DeleteRecursively(log_basedir)
        os.system("rm -rf "+log_basedir)
    tf.gfile.MakeDirs(log_basedir)
    config_info['log_path'] = log_basedir

    if os.path.exists(output_dir):
        pass
    else:
        os.mkdir(output_dir)
    print('Created all folders!')

    '''load dataset'''
    if data_tag == 'callee':
        my_data = dataset.Dataset(data_folder, func_path, embed_path, process_num, embed_dim, max_length, num_classes, tag, embd_type)
    else: # caller
        my_data = dataset_caller.Dataset(data_folder, func_path, embed_path, process_num, embed_dim, max_length, num_classes, tag)
    print('Created the dataset!')

    '''get model id list'''
    # model_id_list = sorted(get_model_id_list(model_dir), reverse=True)
    model_id_list = sorted(get_model_id_list(model_dir))
    with tf.Graph().as_default(), tf.Session() as session:
        # generate placeholder
        data_pl, label_pl, length_pl, keep_prob_pl = placeholder_inputs(num_classes, max_length, embed_dim)
        # generate model
        model = Model(session, my_data, config_info, data_pl, label_pl, length_pl, keep_prob_pl)
        print('Created the model!')

        for model_id in model_id_list:
            result_path = os.path.join(output_dir, 'test_result_%d.pkl' % model_id)
            if os.path.exists(result_path):
                continue
            else:
                pass
            model_path = os.path.join(model_dir, 'model-%d' % model_id)
            model.saver.restore(session, model_path)

            total_result = model.test()
            my_data._index_in_test = 0
            my_data.test_tag = True
            print(total_result['acc'])
            with open(result_path, 'w') as f:
                pickle.dump(total_result, f)
            print('Save the test result !!! ... %s' % result_path)






def get_config():
    '''
    get config information from command line
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_folder', dest='data_folder', help='The data folder of training dataset.', type=str, required=True)
    parser.add_argument('-o', '--output_dir', dest='output_dir', help='The directory to saved the log information & models.', type=str, required=True)
    parser.add_argument('-f', '--split_func_path', dest='func_path', help='The path of file saving the training & testing function names.', type=str, required=True)
    parser.add_argument('-e', '--embed_path', dest='embed_path', help='The path of saved embedding vectors.', type=str, required=True)
    parser.add_argument('-m', '--model_dir', dest='model_dir', help='The directory saved the models.', type=str, required=True)
    parser.add_argument('-t', '--label_tag', dest='tag', help='The type of labels. Possible value: num_args, type#0, type#1, ...', type=str, required=False, default='num_args')
    parser.add_argument('-dt', '--data_tag', dest='data_tag', help='The type of input data.', type=str, required=False, choices=['caller', 'callee'], default='callee')
    parser.add_argument('-pn', '--process_num', dest='process_num', help='Number of processes.', type=int, required=False, default=40)
    parser.add_argument('-ed', '--embedding_dim', dest='embed_dim', help='The dimension of embedding vector.', type=int, required=False, default=256)
    parser.add_argument('-ml', '--max_length', dest='max_length', help='The maximum length of input sequences.', type=int, required=False, default=500)
    parser.add_argument('-nc', '--num_classes', dest='num_classes', help='The number of classes', type=int, required=False, default=16)
    parser.add_argument('-en', '--epoch_num', dest='epoch_num', help='The number of epoch.', type=int, required=False, default=20)
    parser.add_argument('-s', '--save_frequency', dest='save_batchs', help='The frequency for saving the trained model.', type=int, required=False, default=100)
    parser.add_argument('-do', '--dropout', dest='dropout', help='The dropout value.', type=float, required=False, default=0.8)
    parser.add_argument('-nl', '--num_layers', dest='num_layers', help='Number of layers in RNN.', type=int, required=False, default=3)
    parser.add_argument('-ms', '--max_to_save', dest='max_to_save', help='Maximum number of models saved in the directory.', type=int, required=False, default=100)
    parser.add_argument('-b', '--batch_size', dest='batch_size', help='The size of batch.', type=int, required=False, default=256)
    parser.add_argument('-p', '--summary_frequency', dest='summary_frequency', help='The frequency of showing the accuracy & cost value.', type=int, required=False, default=20)

    args = parser.parse_args()
    
    config_info = {
        'data_folder': args.data_folder,
        'output_dir': args.output_dir,
        'func_path': args.func_path,
        'embed_path': args.embed_path,
        'tag': args.tag,
        'model_dir': args.model_dir,
        'data_tag': args.data_tag,
        'process_num': args.process_num,
        'embed_dim': args.embed_dim,
        'max_length': args.max_length,
        'num_classes': args.num_classes,
        'epoch_num': args.epoch_num,
        'save_batchs': args.save_batchs,
        'dropout': args.dropout,
        'num_layers': args.num_layers,
        'max_to_save': args.max_to_save,
        'batch_size': args.batch_size,
        'summary_frequency': args.summary_frequency,
        'embedding_type': args.embedding_type
    }

    return config_info


def main():
    config_info = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    training(config_info)
    testing(config_info)

if __name__ == '__main__':
    main()
