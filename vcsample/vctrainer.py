from __future__ import print_function
import random
import math
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from .classify import Classifier, read_node_label

class vctrainer(object):
    def __init__(self, graph, model_v, model_c, rep_size=128, epoch = 10, batch_size=1000, learning_rate=0.001,
                negative_ratio=5):
        self.cur_epoch = 0
        self.g = graph
        self.model_v = model_v
        self.model_c = model_c
        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
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
        self.get_embeddings()

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
        vs = self.model_v.sample_v(self.batch_size)
        batch_id = 0
        for h in vs:
            t = self.model_c.sample_c(h)
            sign = [1.]
            _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
                                self.h: h, self.t: t, self.sign: sign})
            sum_loss += cur_loss
            batch_id += 1
            for i in range(self.negative_ratio):
                t = self.neg_batch(h)
                sign = [-1.]
                _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
                                self.h: h, self.t: t, self.sign: sign})
                sum_loss += cur_loss
                batch_id += 1

        print('epoch:{} sum of loss:{!s}'.format(self.cur_epoch, sum_loss))
        self.cur_epoch += 1

    def neg_batch(self, h):
        t = []
        for i in range(len(h)):
            t.append(random.randint(0, self.node_size-1))
        return t

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()

