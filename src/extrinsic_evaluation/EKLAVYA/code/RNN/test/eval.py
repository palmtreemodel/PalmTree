import tensorflow as tf
import dataset
import dataset_caller
import os
import sys

import argparse
import functools
import pickle
import inspect


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


def fill_feed_dict(data_set, batch_size, data_tag, keep_prob, data_pl, label_pl, length_pl, keep_prob_pl):
    data_batch = data_set.get_batch(batch_size=batch_size)

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
        self.batch_size = int(config_info['batch_size'])

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

        self.saver = tf.train.Saver(tf.trainable_variables())

        tf.global_variables_initializer().run()

    def test(self):
        total_result = {
            'cost': [],
            'pred': [],
            'func_name': []
        }
        while self.datasets.test_tag:
            feed_dict, func_name_list = fill_feed_dict(self.datasets, self.batch_size, 'test', 1.0,
                                                       self._data, self._label, self._length, self._keep_prob)
            cost_result, pred_result = self.session.run(
                [self.cost_list, self.pred_label],
                feed_dict = feed_dict
            )
            total_result['cost'].append(cost_result)
            total_result['pred'].append(pred_result)
            total_result['func_name'].append(func_name_list)

        return total_result


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

    '''create model & log folder'''
    if os.path.exists(output_dir):
        pass
    else:
        os.mkdir(output_dir)
    print('Created all folders!')

    '''load dataset'''
    if data_tag == 'callee':
        my_data = dataset.Dataset(data_folder, func_path, embed_path, process_num, embed_dim, max_length, num_classes, tag)
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
            with open(result_path, 'w') as f:
                pickle.dump(total_result, f)
            print('Save the test result !!! ... %s' % result_path)


def get_config():
    '''
    get config information
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_folder', dest='data_folder', help='The data folder of testing dataset.', type=str, required=True)
    parser.add_argument('-f', '--split_func_path', dest='func_path', help='The path of file saving the training & testing function names.', type=str, required=True)
    parser.add_argument('-e', '--embed_path', dest='embed_path', help='The path of file saving embedding vectors.', type=str, required=True)
    parser.add_argument('-o', '--output_dir', dest='output_dir', help='The directory to saved the evaluation result.', type=str, required=True)
    parser.add_argument('-m', '--model_dir', dest='model_dir', help='The directory saved the models.', type=str, required=True)
    parser.add_argument('-t', '--label_tag', dest='tag', help='The type of labels. Possible value: num_args, type#0, type#1, ...', type=str, required=False, default='num_args')
    parser.add_argument('-dt', '--data_tag', dest='data_tag', help='The type of input data.', type=str, required=False, choices=['caller', 'callee'], default='callee')
    parser.add_argument('-pn', '--process_num', dest='process_num', help='Number of processes.', type=int, required=False, default=40)
    parser.add_argument('-ed', '--embedding_dim', dest='embed_dim', help='The dimension of embedding vector.', type=int, required=False, default=256)
    parser.add_argument('-ml', '--max_length', dest='max_length', help='The maximun length of input sequences.', type=int, required=False, default=500)
    parser.add_argument('-nc', '--num_classes', dest='num_classes', help='The number of classes', type=int, required=False, default=16)
    parser.add_argument('-do', '--dropout', dest='dropout', help='The dropout value.', type=float, required=False, default=1.0)
    parser.add_argument('-nl', '--num_layers', dest='num_layers', help='Number of layers in RNN.', type=int, required=False, default=3)
    parser.add_argument('-b', '--batch_size', dest='batch_size', help='The size of batch.', type=int, required=False, default=256)

    args = parser.parse_args()
    
    config_info = {
        'data_folder': args.data_folder,
        'func_path': args.func_path,
        'embed_path': args.embed_path,
        'tag': args.tag,
        'data_tag': args.data_tag,
        'process_num': args.process_num,
        'embed_dim': args.embed_dim,
        'max_length': args.max_length,
        'num_classes': args.num_classes,
        'output_dir': args.output_dir,
        'model_dir': args.model_dir,
        'dropout': args.dropout,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size
    }

    return config_info



def main():
    config_info = get_config()
    testing(config_info)


if __name__ == '__main__':
    main()