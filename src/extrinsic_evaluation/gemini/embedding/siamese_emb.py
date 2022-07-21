import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

flags = tf.app.flags
FLAGS = flags.FLAGS

class Siamese:

    #calculate embedding
    def emb_generation(self, x, n):
        mul_mat = x[:, FLAGS.vector_size:]
        x = x[:, :FLAGS.vector_size]
        # tf.reset_default_graph()

        # embeddings to be calculated
        #print("x shape:" , tf.shape(x))
        #print("n shape:" , tf.shape(n))
        #print(self.W_1)

        mu_val = tf.zeros_like(x, name="mu_val")
        # tf.Variable(mu_v, name="mu_val", validate_shape=False, trainable=False)

        # Running T times
        for t in range(FLAGS.T):
            #calculating summation of neighbour vertexes
            mu_val = tf.matmul(n, mu_val, name="neighbour_summation")
            # print("mu_val:", mu_val)
            #non-linear trabsformation
            sig_1  = tf.nn.relu(tf.matmul(mu_val, self.P_1), name="relu_op_lv1")
            sig_2  = tf.nn.relu(tf.matmul(sig_1,  self.P_2), name="relu_op_lv2")
            #new embedding value
            mu_val = tf.nn.tanh(tf.matmul(x, self.W_1) + sig_2, name="new_mu_value")

        #summation across column
        #print("mu_valu shape", tf.shape(mu_val))
        #print(mu_val)
        if mul_mat.shape[1] == 0:
            mu_summ = tf.reduce_sum(mu_val, axis=0, name="cumm_column_sum")
            g_embedding = tf.matmul(tf.reshape(mu_summ,[1,FLAGS.vector_size]), self.W_2,name="embedding_calc")

        else:
            mu_summ = tf.matmul(mul_mat, mu_val, True)
            g_embedding = tf.matmul(mu_summ, self.W_2,name="embedding_calc")

        print("mu_summ shape",tf.shape(mu_summ))
        # print(g_embedding)

        return g_embedding


    #loss function
    def loss_with_spring(self):
        margin = 5.0
        #true labels
        labels_t = self.y_
        #fail labels
        labels_f = tf.subtract(1.0, self.y_, name="1-y_i")
        #calculating eucledian distance
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2),2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # ( y_i * ||CNN(p1_i) - CNN(p2_i)||^2 ) + ( 1-y_i * (max(0, C - ||CNN(p1_i) - CNN(p2_i)||))^2 )
        pos = tf.multiply(labels_t, eucd2, name="y_i_x_eucd2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(0.0, tf.subtract(C, eucd)), 2), name="Ny_i_x_C-eucd_xx_2")
        cumm_losses = tf.add(pos, neg, name="cumm_losses")
        loss = tf.reduce_mean(cumm_losses, name="loss")
        return loss

    #loss funtion with step
    def loss_with_step(self):
        margin = 5.0
        #true labels
        labels_t = self.y_
        #fail labels
        labels_f = tf.subtract(1.0, self.y_, name="1-y_i")
        #calculating eucledian distance
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2),2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0, tf.subtract(C, eucd)), name="Ny_x_C-eucd")
        cumm_losses = tf.add(pos, neg, name="cumm_losses")
        loss = tf.reduce_mean(cumm_losses, name="loss")
        return loss


    def l2_norm(self, x, eps=1e-12):
        return tf.sqrt( tf.reduce_sum(tf.square(x), axis=1) + eps )

    def cosine_norm(self, x, eps=1e-12):
        return tf.sqrt( tf.reduce_sum(tf.matmul(x,tf.transpose(x)), axis=1) + eps )

    def emb_gen(self):
        #creating nn using inputs
        with tf.variable_scope("acfg_embedding") as siam_scope:
            #Left embedding
            self.e1 = self.emb_generation(self.x1, self.n1)
            #print("-->siamese left tensor", self.e1)
            siam_scope.reuse_variables()
            #Right embedding
            self.e2 = self.emb_generation(self.x2, self.n2)

    #siamese cosine loss
    #math
    #[\frac{l \cdot r}{l2_norm(l) \cdot l2_norm(right)}]
    def siamese_cosine_loss(self):
        _y = tf.cast(self.y, tf.float32)
        #trying reset default graph
        #tf.reset_default_graph()
        self.emb_gen()
        #cast true value to float type
        #predict value from left and right tensors using cosine formula
        pred_y = tf.reduce_sum(tf.multiply(self.e1, self.e2) , axis=1)/ (self.cosine_norm(self.e1) * self.cosine_norm(self.e2))
        #print(tf.nn.l2_loss(y-pred)/ tf.cast(tf.shape(left)[0], tf.float32))
        #return tf.nn.l2_loss(y-pred)/ tf.cast(tf.shape(left)[0], tf.float32)
        return tf.nn.l2_loss(pred_y - _y)

        # return self.constrastive_loss(self.e1, self.e2, _y, 0.9)

    #generate embedding of single given acfg
    def get_embedding(self):
        #x
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, None), name="test_x")
        #x neighbours value
        self.shape_x = self.x.get_shape().as_list()
        self.n = tf.placeholder(dtype=tf.float32, shape=(None, self.shape_x[0]), name="test_neighbours_x")

        emb_val = self.emb_generation(self.x, self.n)
        return emb_val

    #Function to be used to predict the similarity between two given ACFG using thier Embedding
    def siamese_pred(self):
        #left embedding
        self.test_e1 = tf.placeholder(dtype=tf.float32, shape=(1, FLAGS.emb_size), name="test_e1")
        #right embedding
        self.test_e2 = tf.placeholder(dtype=tf.float32, shape=(1, FLAGS.emb_size), name="test_e2")
        #predict value from left and right tensors using cosine formula
        pred = tf.reduce_sum(tf.multiply(self.test_e1, self.test_e2) , axis=1)/ (self.cosine_norm(self.test_e1) * self.cosine_norm(self.test_e2))
        return pred

    #constastive loss
    def constrastive_loss(self, left, right, y, margin):
        with tf.name_scope("constrative-loss"):
            d = tf.sqrt(tf.reduce_sum( tf.pow( left-right, 2), axis=1, keep_dims=True))
            tmp = y * tf.square(d)
            tmp2 = (1 - y) *  tf.square( tf.maximum((margin - d), 0))
            return tf.reduce_mean(tmp + tmp2)/2



    #create model
    def __init__(self):
        #with tf.name_scope("input"):
        #	self.n_input = self.nbr_input()
        #input vector/acfg's

        if FLAGS.emb_type != "manual": #and FLAGS.emb_type != "cfg_bert" and FLAGS.emb_type != 'mlm_only':
            with tf.name_scope("basicblocks-rnn"):
                if FLAGS.emb_type == '1hot':
                    self.ins = tf.placeholder(dtype=tf.float32, shape=(1, None), name="input_ins")
                    self.length = tf.placeholder(dtype=tf.float32, shape=(None), name="input_length")
                    self.rnncell = tf.compat.v1.nn.rnn_cell.GRUCell(FLAGS.vector_size)
                    self.embedding = layers.Embedding(5001, FLAGS.vector_size)
                    self.emb_ins =keras.activations.tanh(self.embedding(self.ins))
                else:
                    self.ins = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.vector_size), name="input_ins")
                    self.length = tf.placeholder(dtype=tf.float32, shape=(None), name="input_length")
                    self.rnncell = tf.compat.v1.nn.rnn_cell.GRUCell(FLAGS.vector_size)
                    self.emb_ins = tf.layers.dense(self.ins, FLAGS.vector_size, activation=tf.nn.elu)

                _, self.bb_emb = tf.nn.dynamic_rnn(self.rnncell, self.ins, dtype=tf.float32)


        with tf.name_scope("acfgs-siamese"):
            #Input 1
            self.x1 = tf.placeholder(dtype=tf.float32, shape=(None, FLAGS.vector_size), name="input_x1")
            #Input 2
            self.x2 = tf.placeholder(dtype=tf.float32, shape=(None, FLAGS.vector_size), name="input_x2")
            #Resulting Label
            self.y = tf.placeholder(dtype=tf.int32, name="input_y")
            #x1 neighbour value
            self.shape_x1 = self.x1.get_shape().as_list()
            self.n1 = tf.placeholder(dtype=tf.float32, shape=(None, self.shape_x1[0]), name="neighbours_x1")
            #x2 neighbour value
            self.shape_x2 = self.x2.get_shape().as_list()
            self.n2 = tf.placeholder(dtype=tf.float32, shape=(None, self.shape_x2[0]), name="neighbours_x2")

            w_init = tf.truncated_normal_initializer(stddev=0.1)
            #learnable parameters
            self.W_1 = tf.get_variable(name='W_1', dtype=tf.float32, shape=(FLAGS.vector_size, FLAGS.vector_size), initializer=w_init)
            self.W_2 = tf.get_variable(name='W_2', dtype=tf.float32, shape=(FLAGS.vector_size, FLAGS.emb_size), initializer=w_init)
            self.P_1 = tf.get_variable(name='P1_relu', dtype=tf.float32, shape=(FLAGS.vector_size, FLAGS.vector_size), initializer=w_init)
            self.P_2 = tf.get_variable(name='P2_relu', dtype=tf.float32, shape=(FLAGS.vector_size, FLAGS.vector_size), initializer=w_init)

        with tf.name_scope("loss"):
            self.loss = self.siamese_cosine_loss()
        #create loss

