from __future__ import print_function
import random
import math
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from .classify import Classifier, read_node_label

class vctrainer(object):
    # ngmode=0: uniform; ngmode=1: power 0.75
    def __init__(self, graph, model, rep_size=128, epoch = 10, learning_rate=0.001,
                negative_ratio=5, ngmode = 0, label_file=None, clf_ratio=0.5, auto_save=True):
        self.cur_epoch = 0
        self.g = graph
        self.model = model
        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.ngmode = ngmode
        if negative_ratio > 0 and ngmode == 1:
            self.gen_negative(table_size=1e6)
        self.sess = tf.Session()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False, seed=cur_seed)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.build_model()
        self.sess.run(tf.global_variables_initializer())
        print("Start training.")
        for i in range(epoch):
            self.train_one_epoch()
            if label_file:
                self.get_embeddings()
                X, Y = read_node_label(label_file)
                print("Training classifier using {:.2f}% nodes...".format(
                    clf_ratio*100))
                clf = Classifier(vectors=self.vectors,
                                 clf=LogisticRegression())
                result = clf.split_train_evaluate(X, Y, clf_ratio)

                if result['macro'] > self.best_result:
                    self.best_result = result['macro']
                    if auto_save:
                        self.best_vector = self.vectors

        self.get_embeddings()
        if auto_save and label_file:
            self.vectors = self.best_vector

    def get_embeddings(self):
        vectors = {}
        embeddings = self.embeddings.eval(session=self.sess)
        # embeddings = self.sess.run(tf.nn.l2_normalize(self.embeddings.eval(session=self.sess), 1))
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        self.vectors = vectors

    def build_model(self):
        self.h = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.sign = tf.placeholder(tf.float32, [None])

        cur_seed = random.getrandbits(32)
        self.embeddings = tf.get_variable(name="embeddings", shape=[
                                          self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        self.context_embeddings = tf.get_variable(name="context_embeddings", shape=[
                                                  self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        # self.h_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.h), 1)
        # self.t_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.t), 1)
        # self.t_e_context = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.context_embeddings, self.t), 1)
        self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
        # self.t_e = tf.nn.embedding_lookup(self.embeddings, self.t)
        self.t_e_context = tf.nn.embedding_lookup(
            self.context_embeddings, self.t)
        self.loss = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(
            self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)),1e-8,1.0)))
        # self.loss = -tf.reduce_mean(tf.log_sigmoid(
        #     self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def train_one_epoch(self):
        sum_loss = 0.0
        batches = self.model.batch_iter()
        batch_id = 0
        for batch in batches:
            h, t, sign = batch
            _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
                                self.h: h, self.t: t, self.sign: sign})
            sum_loss += cur_loss
            batch_id += 1
            for i in range(self.negative_ratio):
                t, sign = self.neg_batch(h)
                _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
                                self.h: h, self.t: t, self.sign: sign})
                sum_loss += cur_loss
                batch_id += 1

        print('epoch:{} sum of loss:{!s}'.format(self.cur_epoch, sum_loss))
        self.cur_epoch += 1

    def gen_negative(self, table_size=1e8, power=0.75):
        # table for negative sampling
        self.table_size = table_size
        numNodes = self.node_size

        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = np.zeros(numNodes)

        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            node_degree[look_up[edge[0]]
                        ] += self.g.G[edge[0]][edge[1]]["weight"]

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1

    def neg_batch(self, h):
        sign = -1.
        t = []
        for i in range(len(h)):
            if self.ngmode == 1:
                t.append(
                    self.sampling_table[random.randint(0, self.table_size-1)])
            else:
                t.append(random.randint(0, numNodes-1))
        return t, [sign]

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()

