from __future__ import print_function
import tensorflow as tf
import time

import threading

from six.moves import xrange

class vctrainer(object):
    def __init__(self,
                 graph,
                 vcsampler, emb_model, emb_file,
                 rep_size=128, epoch=10, batch_size=1000, learning_rate=0.2, negative_ratio=5, thread_num=1):
        self.g = graph
        self.vc_model= vcsampler
        self.emb_model = emb_model #sym, asym
        self.node_size = graph.G.number_of_nodes()
        self.node_degree_dist = graph.degree_dist
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.negative_ratio = negative_ratio
        self.emb_file = emb_file
        self.num_of_epochs = epoch

        self.words_to_train = self.vc_model.pair_per_epoch() * self.num_of_epochs
        # print("total words to train {}.".format(self.words_to_train))
        self._session = tf.Session()
        self.build_graph()
        self.train()  # Call train function directly.
        self.get_embeddings()

    def get_embeddings(self):
        vectors = {}
        embeddings = self.embeddings.eval(session=self._session)
        # embeddings = self.sess.run(tf.nn.l2_normalize(self.embeddings.eval(session=self.sess), 1))
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        self.vectors = vectors

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()

    def forward(self, examples, labels):
        """Build the graph for the forward pass."""

        # Declare all variables we need.
        # Embedding: [node_size, emb_dim]
        init_width = 0.5 / self.rep_size
        emb = tf.Variable(
            tf.random_uniform(
                [self.node_size, self.rep_size], -init_width, init_width),
            name="emb")
        self.embeddings = emb

        # Softmax weight: [node_size, emb_dim]. Transposed. TODO: this is only asymmetry model!!!
        sm_w_t = tf.Variable(
            tf.zeros([self.node_size, self.rep_size]),
            name="sm_w_t")

        # Softmax bias: [node_size].
        sm_b = tf.Variable(tf.zeros([self.node_size]), name="sm_b")

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [self.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=self.negative_ratio,
            unique=True,
            range_max=self.node_size,
            distortion=0.75,
            unigrams=self.node_degree_dist))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)

        # Weights for labels: [batch_size, emb_dim]
        true_w = tf.nn.embedding_lookup(sm_w_t, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(sm_b, labels)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w = tf.nn.embedding_lookup(sm_w_t, sampled_ids)
        # Biases for sampled ids: [num_sampled, 1]
        sampled_b = tf.nn.embedding_lookup(sm_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_b_vec = tf.reshape(sampled_b, [self.negative_ratio])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec
        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / self.batch_size
        return nce_loss_tensor

    def optimize(self, loss):
        """Build the graph to optimize the loss function."""

        # Optimizer nodes.
        # Linear learning rate decay.
        words_to_train = self.words_to_train
        lr = self.learning_rate * tf.maximum(
            0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
        self._lr = lr
        optimizer = tf.train.GradientDescentOptimizer(lr)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss,
                                   global_step=self.global_step,
                                   gate_gradients=optimizer.GATE_NONE)
        self._train = train


    def build_graph(self):
        """Build the graph for the full model."""
        self._examples = tf.placeholder(tf.int32, [None])
        self._labels = tf.placeholder(tf.int32, [None])
        self._words = tf.placeholder(tf.float32, shape=())

        true_logits, sampled_logits = self.forward(self._examples, self._labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        # tf.summary.scalar("NCE loss", loss)
        self._loss = loss
        self.optimize(loss)

        # Properly initialize all variables.
        self._session.run(tf.global_variables_initializer())

    def train(self):
        words = 0.0
        for epoch in xrange(self.num_of_epochs):
            sum_loss = 0.0
            tot_time = 0.0
            batch_id = 0
            start = time.time()
            for batch in self.vc_model.generate_batch(self.batch_size):
                examples, labels = batch
                words = words + len(examples)
                tx = time.time()
                _, cur_loss = self._session.run([self._train, self._loss], feed_dict={
                self._examples: examples, self._labels: labels, self._words: words})
                tot_time += time.time() - tx
                sum_loss += cur_loss
                batch_id = batch_id + 1  # tf.train.get_global_step()
            end = time.time()
            if (epoch + 1) % 5 == 0:
                self.get_embeddings()
                self.save_embeddings(self.emb_file+"_"+str(epoch + 1))
            print('epoch {}: sum of loss:{:.8f}; time cost: {:.3f}/{:.3f}, per_batch_cost: {:.3f}, '
                  'words_trained {:.0f}/{:.0f}'.
                  format(epoch, sum_loss / batch_id, tot_time, end-start, tot_time/batch_id, words/self.batch_size, self.words_to_train/self.batch_size))

    # def _train_thread_body(self):
    #     initial_epoch, = self._session.run([self._epoch])
    #     while True:
    #         _, epoch = self._session.run([self._train, self._epoch])
    #         if epoch != initial_epoch:
    #             break
    #
    # def train(self):
    #     """Train the model."""
    #     # opts = self._options
    #
    #     initial_epoch, initial_words = self._session.run([self._epoch, self._words])
    #
    #     # summary_op = tf.summary.merge_all()
    #     # summary_writer = tf.summary.FileWriter(opts.save_path, self._session.graph)
    #     workers = []
    #     for _ in xrange(opts.concurrent_steps):
    #         t = threading.Thread(target=self._train_thread_body)
    #         t.start()
    #         workers.append(t)
    #
    #     # last_words, last_time, last_summary_time = initial_words, time.time(), 0
    #     # last_checkpoint_time = 0
    #     # while True:
    #     #     time.sleep(opts.statistics_interval)  # Reports our progress once a while.
    #     #     (epoch, step, loss, words, lr) = self._session.run(
    #     #         [self._epoch, self.global_step, self._loss, self._words, self._lr])
    #     #     now = time.time()
    #     #     last_words, last_time, rate = words, now, (words - last_words) / (
    #     #             now - last_time)
    #     #     print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
    #     #           (epoch, step, lr, loss, rate), end="")
    #     #     sys.stdout.flush()
    #     #     # if now - last_summary_time > opts.summary_interval:
    #     #     #     summary_str = self._session.run(summary_op)
    #     #     #     summary_writer.add_summary(summary_str, step)
    #     #     #     last_summary_time = now
    #     #     # if now - last_checkpoint_time > opts.checkpoint_interval:
    #     #     #     self.saver.save(self._session,
    #     #     #                     os.path.join(opts.save_path, "model.ckpt"),
    #     #     #                     global_step=step.astype(int))
    #     #     #     last_checkpoint_time = now
    #     #     if epoch != initial_epoch:
    #     #         break
    #
    #     for t in workers:
    #         t.join()
    #
    #     return epoch


    # def build_model(self, emb_model):
    #     self.h = tf.placeholder(tf.int32, [None])
    #     self.t = tf.placeholder(tf.int32, [None])
    #     self.sign = tf.placeholder(tf.float32, [None])
    #
    #     cur_seed = random.getrandbits(32)
    #     if emb_model == 'asym':
    #         print("using asym loss!")
    #         self.embeddings = tf.get_variable(name="embeddings", shape=[
    #                                       self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
    #         self.context_embeddings = tf.get_variable(name="context_embeddings", shape=[
    #                                               self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
    #         self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
    #         self.t_e = tf.nn.embedding_lookup(self.context_embeddings, self.t) #context emb, second order loss
    #     else:
    #         print("using sym loss!")
    #         self.embeddings = tf.get_variable(name="embeddings", shape=[
    #             self.node_size, self.rep_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
    #         self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
    #         self.t_e = tf.nn.embedding_lookup(self.embeddings, self.t) #word_emb, first order loss
    #
    #     # self.loss = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(
    #     #     self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)),1e-8,1.0))) # why use clip?
    #     # self.loss = -tf.reduce_mean(tf.log_sigmoid(
    #     #     self.sign*tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)))
    #     logits = tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)
    #     self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.sign))
    #     optimizer = tf.train.AdamOptimizer(self.learning_rate)
    #     self.train_op = optimizer.minimize(self.loss)
    #
    # """
    #    positive and negative batches seperately.
    # """
    # # def train_one_epoch(self):
    # #     sum_loss = 0.0
    # #
    # #     batch_id = 0
    # #     tot_time = 0.0
    # #     start = time.time()
    # #     first = True
    # #     for batch in self.model_v.sample_batch(self.batch_size, self.negative_ratio):
    # #         h1, t1, sign = batch
    # #         # sign = [1.0] #for _ in range(len(h1))]
    # #         if first:
    # #             first = False
    # #             print("real batch size={}".format(len(h1)))
    # #         tx = time.time()
    # #         _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
    # #                             self.h: h1, self.t: t1, self.sign: sign})
    # #         # cur_loss = 0.0
    # #         # print(len(h1))
    # #         tot_time += time.time() - tx
    # #         sum_loss += cur_loss
    # #         batch_id += 1 #positive batch
    # #         # for i in range(self.negative_ratio):
    # #         #     t1 = self.neg_batch(h1)
    # #         #     sign = [0.0] # for _ in range(len(h1))]
    # #         #     tx = time.time()
    # #         #     _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
    # #         #                     self.h: h1, self.t: t1, self.sign: sign})
    # #         #     tot_time += time.time() - tx
    # #         #     sum_loss += cur_loss
    # #         #     # print('\tBatch {}: loss:{!s}/{!s}'.format(batch_id, cur_loss, sum_loss))
    # #         #     batch_id += 1
    # #     end = time.time()
    # #
    # #     print('epoch {}: sum of loss:{!s}; time cost: {!s}/{!s}, per_batch_cost: {!s}'.
    # #           format(self.cur_epoch, sum_loss / batch_id, tot_time, end-start, tot_time/batch_id))
    #
    # """
    # mix negative batch and positive batches
    # """
    # def train_one_epoch(self):
    #     sum_loss = 0.0
    #
    #     batch_id = 0
    #
    #     tot_time = 0.0
    #     start = time.time()
    #     for batch in self.model_v.sample_batch(self.batch_size, self.negative_ratio):
    #         h1, t1, sign = batch
    #         tx = time.time()
    #         _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
    #                             self.h: h1, self.t: t1, self.sign: sign})
    #         tot_time += time.time() - tx
    #         sum_loss += cur_loss
    #         batch_id += 1
    #     end = time.time()
    #
    #     print('epoch {}: sum of loss:{!s}; time cost: {!s}/{!s}, per_batch_cost: {!s}'.
    #           format(self.cur_epoch, sum_loss / batch_id, tot_time, end-start, tot_time/batch_id))
    #
    # # def train_one_epoch1(self):
    # #     sum_loss = 0.0
    # #     vs = self.model_v.sample_v(self.batch_size)
    # #     batch_id = 0
    # #     for hx in vs:
    # #         # TODO, return two lists
    # #         h1, t1 = self.model_c.sample_c(hx)
    # #         sign = [1.]
    # #         _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
    # #                             self.h: h1, self.t: t1, self.sign: sign})
    # #         sum_loss += cur_loss
    # #         batch_id += 1 #positive batch
    # #         for i in range(self.negative_ratio):
    # #             t1 = self.neg_batch(h1)
    # #             sign = [-1.]
    # #             _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict={
    # #                             self.h: h1, self.t: t1, self.sign: sign})
    # #             sum_loss += cur_loss
    # #             # print('\tBatch {}: loss:{!s}/{!s}'.format(batch_id, cur_loss, sum_loss))
    # #             batch_id += 1
    # #
    # #     print('epoch {}: sum of loss:{!s}'.format(self.cur_epoch, sum_loss / batch_id))
    #
    # def neg_batch(self, h):
    #     t = []
    #     for i in range(len(h)):
    #         t.append(random.randint(0, self.node_size-1))
    #     return t



