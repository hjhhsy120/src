from __future__ import print_function
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from .graph import *
from .classify import Classifier, read_node_label
import time

import pickle
from sklearn import metrics, model_selection, pipeline
from sklearn.preprocessing import StandardScaler
import os
# import tensorflow as tf

from .getmodel import getmodels

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
    "concat": lambda a, b: np.concatenate([a, b])
}


def create_train_test_graphs(args):
    """
    Create and cache train & test graphs.
    Will load from cache if exists unless --regen option is given.

    :param args:
    :return:
        Gtrain, Gtest: Train & test graphs
    """
    # Remove half the edges, and the same number of "negative" edges

    # Create random training and test graphs with different random edge selections

    Gtrain = None
    cached_fn = args.cached_fn
    if cached_fn != '':
        if os.path.exists(cached_fn):
            print("Loading link prediction graphs from %s" % cached_fn)
            with open(cached_fn, 'rb') as f:
                cache_data = pickle.load(f)
            Gtrain = cache_data['g_train']
    if Gtrain is None:
        print("Regenerating link prediction graphs")
        # Train graph embeddings on graph with random links
        Gtrain = Graph(prop_pos=args.prop_pos,
                          prop_neg=args.prop_neg,
                          prop_neg_tot=args.prop_neg_tot)
        if args.graph_format == 'adjlist':
            Gtrain.read_adjlist(filename=args.input)
        elif args.graph_format == 'edgelist':
            Gtrain.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed)
        Gtrain.generate_pos_neg_links()

        # Cache generated  graph
        if cached_fn != '':
            cache_data = {'g_train': Gtrain}
            with open(cached_fn, 'wb') as f:
                pickle.dump(cache_data, f)

    return Gtrain

def edges_to_features(vectors, edge_list, edge_function, dimensions):
    n_tot = len(edge_list)
    feature_vec = np.empty((n_tot, dimensions), dtype='f')

    # Iterate over edges
    for ii in range(n_tot):
        v1, v2 = edge_list[ii]

        # Edge-node features
        emb1 = np.asarray(vectors[str(v1)])
        emb2 = np.asarray(vectors[str(v2)])

        # Calculate edge feature
        feature_vec[ii] = edge_function(emb1, emb2)

    return feature_vec
'''
def batch_iter(vectors, edges, labels, batch_size):
    tot = len(labels)
    idx = np.random.permutation(tot)
    v1s = []
    v2s = []
    ls = []
    for i in range(tot):
        v1s += [vectors[edges[idx[i]][0]]]
        v2s += [vectors[edges[idx[i]][1]]]
        ls += [labels[idx[i]]]
        if (i + 1) % batch_size == 0:
            yield v1s, v2s, ls
            v1s = []
            v2s = []
            ls = []
    if ls != []:
        yield v1s, v2s, ls

def full_batch(vectors, edges, labels):
    tot = len(labels)
    idx = np.random.permutation(tot)
    v1s = []
    v2s = []
    ls = []
    for i in range(tot):
        v1s += [vectors[edges[idx[i]][0]]]
        v2s += [vectors[edges[idx[i]][1]]]
        ls += [labels[idx[i]]]
    return v1s, v2s, ls
'''
def test_edge_functions(args):
    dims = args.representation_size
    t1 = time.time()
    print("Reading...")
    Gtrain = create_train_test_graphs(args)

    # Train and test graphs, with different edges
    edges_test, labels_test = Gtrain.get_test_edges()
    edges_train, labels_train = Gtrain.get_train_edges()

    # With fixed test & train graphs (these are expensive to generate)
    # we perform k iterations of the algorithm
    # TODO: It would be nice if the walks had a settable random seed
    aucs = {name: [] for name in edge_functions}

    # Learn embeddings with current parameter values
    model = getmodels(Gtrain, args)

    t2 = time.time()
    print("time: {}".format(t2-t1))
    vectors = model.vectors
    '''
    # tensorflow for (v1)^t W v2
    cur_seed = random.getrandbits(32)
    v1 = tf.placeholder(tf.float32, [None, dims])
    v2 = tf.placeholder(tf.float32, [None, dims])
    y = tf.placeholder(tf.float32, [None])
    w = tf.get_variable(name="w", shape=[
                dims, dims], initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
    v1w = tf.matmul(v1, w)
    score = tf.sigmoid(tf.reduce_mean(tf.multiply(v1w, v2), 1))
    loss = tf.reduce_mean(tf.square(score - y))
    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)
    auc_value, auc_op = tf.metrics.auc(y, score)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    allv1s, allv2s, allls = full_batch(vectors, edges_train, labels_train)
    d = {v1: allv1s, v2: allv2s, y:allls}
    cnt = 0
    print(len(allls))
    precost = 1.
    for i in range(100):
        for v1s, v2s, ls in batch_iter(vectors, edges_train, labels_train, 100):
            sess.run(train, feed_dict = {v1: v1s, v2: v2s, y: ls})
            cnt += 1
        if (i + 1) % 10 == 0:
            cost = sess.run(loss, feed_dict = d)
            print("epoch {} (iter {}): loss = {}".format(i+1, cnt, cost))
            if precost - cost < 0.001:
                break
            precost = cost
    sess.run(auc_op, feed_dict = d)
    auc = sess.run(auc_value, feed_dict = d)
    print("AUC: {}".format(auc))
    return auc
    '''
    for edge_fn_name, edge_fn in edge_functions.items():
        # Calculate edge embeddings using binary function
        dim2 = dims
        if edge_fn_name == 'concat':
            dim2 = dims * 2
        edge_features_train = edges_to_features(vectors, edges_train, edge_fn, dim2)
        edge_features_test = edges_to_features(vectors, edges_test, edge_fn, dim2)

        # Linear classifier
        scaler = StandardScaler()
        lin_clf = LogisticRegression(C=1)
        clf = pipeline.make_pipeline(scaler, lin_clf)

        # Train classifier
        clf.fit(edge_features_train, labels_train)
        auc_train = metrics.scorer.roc_auc_scorer(clf, edge_features_train, labels_train)
        auc_test = metrics.scorer.roc_auc_scorer(clf, edge_features_test, labels_test)

        print("%s -- AUC train: %.4g AUC test: %.4g"
              % (edge_fn_name, auc_train, auc_test))
        aucs[edge_fn_name].append(auc_test)

    print("Edge function test performance (AUC):")
    for edge_name in aucs:
        auc_mean = np.mean(aucs[edge_name])
        auc_std = np.std(aucs[edge_name])
        print("[%s] mean: %.4g +/- %.3g" % (edge_name, auc_mean, auc_std))

    for edge_name in aucs:
        auc_mean = np.mean(aucs[edge_name])
        print("%.4g\t" % (auc_mean), end='')

    return aucs

