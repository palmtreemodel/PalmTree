'''
train the embedding model in order to get the vectors representing each instructions
the vectors embed the semantic information of instruction inside

the input: the output of prep_embed_input (output_path)
the output: the embedding model and mapping between the integer to vectors
'''

import os
import sys
import argparse
import tensorflow as tf
import threading
import pickle
import time
from tensorflow.models.embedding import gen_word2vec as word2vec
from six.moves import xrange

class Options(object):
    def __init__(self):
        pass


class TrainEmbed(object):
    def __init__(self, args, session):
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.global_epoch=0
        self.num_threads = int(args['thread_num'])
        self.data_file_path = args['input_path']
        self.read_config(args)
        self.get_output_path(args)
        self.build_graph()


    def read_config(self, config):
        self._options = Options()
        self._options.train_data = config['input_path']
        self._options.emb_dim = int(config['embedding_size'])
        self._options.batch_size = int(config['batch_size'])
        self._options.window_size = int(config['window_size'])
        self._options.min_count = int(config['min_count'])
        self._options.subsample = float(config['subsample'])
        self._options.epochs_to_train = int(config['num_epochs'])
        self._options.learning_rate = float(config['learning_rate'])
        self._options.num_samples = int(config['num_neg_samples'])


    def get_output_path(self, config):
        if os.path.isdir(config['output_dir']):
            '''embedding mapping path'''
            cnt = 1
            self.output_path = os.path.join(config['output_dir'], 'embed_%d.emb' % cnt)
            while(os.path.exists(self.output_path)):
                cnt += 1
                self.output_path = os.path.join(config['output_dir'], 'embed_%d.emb' % cnt)
            '''folder for saving embedding model'''
            self.model_folder = os.path.join(config['output_dir'], 'model_%d' % cnt)
            os.mkdir(self.model_folder)
        else:
            error_str = '[ERROR] the output folder does not exist! ... %s' % config['output_dir']
            sys.exit(error_str)


    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        while True:
            _, epoch = self._session.run([self._train, self._epoch])
            if epoch != initial_epoch:
                break


    def build_graph(self):
        """Build the model graph."""
        opts = self._options

        # The training data. A text file.
        (words, counts, words_per_epoch, current_epoch, total_words_processed,
         examples, labels) = word2vec.skipgram(filename=opts.train_data,
                                               batch_size=opts.batch_size,
                                               window_size=opts.window_size,
                                               min_count=opts.min_count,
                                               subsample=opts.subsample)
        (opts.vocab_words, opts.vocab_counts,
         opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)

        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
            self._word2id[w] = i

        # Declare all variables we need.
        # Input words embedding: [vocab_size, emb_dim]
        w_in = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size,
                 opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
            name="w_in")

        # Global step: scalar, i.e., shape [].
        w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")

        # Global step: []
        global_step = tf.Variable(0, name="global_step")

        # Linear learning rate decay.
        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        lr = opts.learning_rate * tf.maximum(
            0.0001,
            1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

        # Training nodes.
        inc = global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            train = word2vec.neg_train(w_in,
                                       w_out,
                                       examples,
                                       labels,
                                       lr,
                                       vocab_count=opts.vocab_counts.tolist(),
                                       num_negative_samples=opts.num_samples)

        self._w_in = w_in
        self._examples = examples
        self._labels = labels
        self._lr = lr
        self._train = train
        self.step = global_step
        self._epoch = current_epoch
        self._words = total_words_processed

        # Properly initialize all variables.
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(max_to_keep=100,)


    def train(self):
        """Train the model."""
        opts = self._options

        initial_epoch, initial_words = self._session.run([self._epoch, self._words])

        workers = []
        for _ in xrange(self.num_threads):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)

        last_words, last_time = initial_words, time.time()
        while True:
            time.sleep(5)  # Reports our progress once a while.
            (epoch, step, words,
             lr) = self._session.run([self._epoch, self.step, self._words, self._lr])
            self.global_epoch = epoch
            now = time.time()
            last_words, last_time, rate = words, now, (words - last_words) / (now - last_time)
            print("Epoch %4d Step %8d: lr = %12.10f words/sec = %8.0f" % (epoch, step, lr, rate))
            sys.stdout.flush()
            if epoch != initial_epoch:
                break

        for t in workers:
            t.join()


    def save(self):
        # save the model and the corresponding training parameters
        insn_embed = {}

        # with tempfile.NamedTemporaryFile() as temp_file:
        ckpt_name = os.path.join(self.model_folder, 'model_%d.ckpt' % self.global_epoch)
        self.saver.save(self._session, ckpt_name)

        insn_embed['vocab_size'] = self._options.vocab_size
        insn_embed['embedding_size'] = self._options.emb_dim
        insn_embed['word2id'] = self._word2id
        insn_embed['id2word'] = self._id2word
        insn_embed['num_epochs'] = self._options.epochs_to_train
        insn_embed['learning_rate'] = self._options.learning_rate
        insn_embed['num_neg_samples'] = self._options.num_samples
        insn_embed['batch_size'] = self._options.batch_size
        insn_embed['window_size'] = self._options.window_size
        insn_embed['min_count'] = self._options.min_count
        insn_embed['subsample'] = self._options.subsample

        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        else:
            pass
        pickle.dump(insn_embed, open(self.output_path, 'wb'))
        print('Saved word embedding network as %s.' % self.output_path)


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', dest='input_path', help='The input file for training embedding model', type=str, required=True)
    parser.add_argument('-o', '--output_dir', dest='output_dir', help='The output folder saving the trained embedding information', type=str, required=False, default='embed_output')
    parser.add_argument('-tn', '--thread_num', dest='thread_num', help='Number of threads', type=int, required=False, default=40)
    parser.add_argument('-sw', '--save_window', dest='save_window', help='Saving the trained information every save_window epoch', type=int, required=False, default=5)
    parser.add_argument('-e', '--embed_dim', dest='embed_dim', help='Dimension of the embedding vector for each instruction', type=int, required=False, default=256)
    parser.add_argument('-ne', '--num_epochs', dest='num_epochs', help='Number of epochs for training the embedding model', type=int, required=False, default=100)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', help='Learning rate', type=float, required=False, default=0.001)
    parser.add_argument('-nn', '--num_neg_smaples', dest='num_neg_samples', help='Number of negative samples', type=int, required=False, default=25)
    parser.add_argument('-b', '--batch_size', dest='batch_size', help='Batch size', type=int, required=False, default=512)
    parser.add_argument('-ws', '--window_size', dest='window_size', help='Window size', type=int, required=False, default=5)
    parser.add_argument('-mc', '--min_count', dest='min_count', help='Ignoring all words with total frequency lower than this', type=int, required=False, default=1)
    parser.add_argument('-s', '--subsample', dest='subsample', help='Subsampling threshold', type=float, required=False, default=0.01)

    args = parser.parse_args()

    config_info = {
        'input_path': args.input_path,
        'output_dir': args.output_dir,
        'thread_num': args.thread_num,
        'save_window': args.save_window,
        'embedding_size': args.embed_dim,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'num_neg_samples': args.num_neg_samples,
        'batch_size': args.batch_size,
        'window_size': args.window_size,
        'min_count': args.min_count,
        'subsample': args.subsample
    }

    return config_info


def main():
    config_info = get_config()

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            '''create the training graph for embedding model'''
            my_embed = TrainEmbed(config_info, session)
            for _ in xrange(my_embed._options.epochs_to_train):
                my_embed.train()
                if my_embed.global_epoch % int(config_info['save_window']) == 0 :
                    my_embed.save()
                else:
                    pass
            my_embed.save()


if __name__ == '__main__':
    main()