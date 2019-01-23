from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from .graph import *
from .classify import Classifier, read_node_label
from .grarep import GraRep
import time

from . import app
from . import vctrainer
from . import deepwalk

import pickle
from sklearn import metrics, model_selection, pipeline
from sklearn.preprocessing import StandardScaler
import os

default_params = {
    'log2p': 0,                     # Parameter p, p = 2**log2p
    'log2q': 0,                     # Parameter q, q = 2**log2q
    'log2d': 7,                     # Feature size, dimensions = 2**log2d
    'num_walks': 10,                # Number of walks from each node
    'walk_length': 80,              # Walk length
    'window_size': 10,              # Context size for word2vec
    'edge_function': "hadamard",    # Default edge function to use
    "prop_pos": 0.5,                # Proportion of edges to remove nad use as positive samples
    "prop_neg": 0.5,                # Number of non-edges to use as negative samples
                                    #  (as a proportion of existing edges, same as prop_pos)
}

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output',
                        help='Output representation file')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--epochs', default=10, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--method', required=True, choices=[
        'deepWalk',
        'app'
    ], help='The learning method')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    parser.add_argument('--feature-file', default='',
                        help='The file of node features')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='the negative ratio of LINE')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    parser.add_argument('--no-auto-save', action='store_true',
                        help='no save the best embeddings when training LINE')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--hidden', default=16, type=int,
                        help='Number of units in hidden layer 1')
    parser.add_argument('--kstep', default=4, type=int,
                        help='Use k-step transition probability matrix')
    parser.add_argument('--lamb', default=0.2, type=float,
                        help='lambda is a hyperparameter in TADW')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=1e-6, type=float,
                        help='alhpa is a hyperparameter in SDNE')
    parser.add_argument('--beta', default=5., type=float,
                        help='beta is a hyperparameter in SDNE')
    parser.add_argument('--nu1', default=1e-5, type=float,
                        help='nu1 is a hyperparameter in SDNE')
    parser.add_argument('--nu2', default=1e-4, type=float,
                        help='nu2 is a hyperparameter in SDNE')
    parser.add_argument('--bs', default=200, type=int,
                        help='batch size of SDNE')
    parser.add_argument('--encoder-list', default='[1000, 128]', type=str,
                        help='a list of numbers of the neuron at each encoder layer, the last number is the '
                             'dimension of the output node representation')
    parser.add_argument('--link-prediction', action='store_true',
                        help='Link prediction task.')
    args = parser.parse_args()

    if args.method != 'gcn' and not args.output:
        print("No output filename. Exit.")
        exit(1)

    return args

def create_train_test_graphs(args):
    """
    Create and cache train & test graphs.
    Will load from cache if exists unless --regen option is given.

    :param args:
    :return:
        Gtrain, Gtest: Train & test graphs
    """
    # Remove half the edges, and the same number of "negative" edges
    prop_pos = default_params['prop_pos']
    prop_neg = default_params['prop_neg']

    # Create random training and test graphs with different random edge selections
    cached_fn = "%s.graph" % (os.path.basename(args.input))
    if os.path.exists(cached_fn):
        print("Loading link prediction graphs from %s" % cached_fn)
        with open(cached_fn, 'rb') as f:
            cache_data = pickle.load(f)
        Gtrain = cache_data['g_train']
    else:
        print("Regenerating link prediction graphs")
        # Train graph embeddings on graph with random links
        Gtrain = Graph(prop_pos=prop_pos,
                          prop_neg=prop_neg)
        if args.graph_format == 'adjlist':
            Gtrain.read_adjlist(filename=args.input)
        elif args.graph_format == 'edgelist':
            Gtrain.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed)
        Gtrain.generate_pos_neg_links()

        # Cache generated  graph
        cache_data = {'g_train': Gtrain}
        with open(cached_fn, 'wb') as f:
            pickle.dump(cache_data, f)

    return Gtrain

def edges_to_features(model, edge_list, edge_function, dimensions):
    n_tot = len(edge_list)
    feature_vec = np.empty((n_tot, dimensions), dtype='f')

    # Iterate over edges
    for ii in range(n_tot):
        v1, v2 = edge_list[ii]

        # Edge-node features
        emb1 = np.asarray(model.vectors[str(v1)])
        emb2 = np.asarray(model.vectors[str(v2)])

        # Calculate edge feature
        feature_vec[ii] = edge_function(emb1, emb2)

    return feature_vec

def test_edge_functions(args):
    t1 = time.time()
    print("Reading...")
    Gtrain = create_train_test_graphs(args)

    # Train and test graphs, with different edges
    edges_train, labels_train = Gtrain.get_selected_edges()
    # edges_test, labels_test = Gtest.get_selected_edges()

    # With fixed test & train graphs (these are expensive to generate)
    # we perform k iterations of the algorithm
    # TODO: It would be nice if the walks had a settable random seed
    aucs = {name: [] for name in edge_functions}

    # Learn embeddings with current parameter values
    if args.method == 'deepWalk':
        model = deepwalk.deepwalk(graph=Gtrain, window=args.window_size)
    elif args.method == 'app':
        model = app.APP(graph=Gtrain)
    trainer = vctrainer.vctrainer(Gtrain, model, model, rep_size=args.representation_size, epoch=args.epochs,
                                    batch_size=1000, learning_rate=args.lr, negative_ratio=args.negative_ratio,
                                    ngmode=1, label_file=None, clf_ratio=args.clf_ratio, auto_save=True)
    model = trainer
    for edge_fn_name, edge_fn in edge_functions.items():
        # Calculate edge embeddings using binary function
        edge_features_train = edges_to_features(model, edges_train, edge_fn, args.representation_size)

        # Linear classifier
        scaler = StandardScaler()
        lin_clf = LogisticRegression(C=1)
        clf = pipeline.make_pipeline(scaler, lin_clf)

        # Train classifier
        clf.fit(edge_features_train, labels_train)
        auc_train = metrics.scorer.roc_auc_scorer(clf, edge_features_train, labels_train)

        aucs[edge_fn_name].append(auc_train)

    print("Edge function test performance (AUC):")
    for edge_name in aucs:
        auc_mean = np.mean(aucs[edge_name])
        auc_std = np.std(aucs[edge_name])
        print("[%s] mean: %.4g +/- %.3g" % (edge_name, auc_mean, auc_std))

    return aucs

def main(args):
    if args.link_prediction:
        test_edge_functions(args)
        return
    t1 = time.time()
    print("Reading...")
    g = Graph()
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted,
                    directed=args.directed)
    if args.method == 'deepWalk':
        model = deepwalk.deepwalk(graph=g, window=args.window_size)
    elif args.method == 'app':
        model = app.APP(graph=g)

    if args.method in ['deepWalk', 'app']:
        trainer = vctrainer.vctrainer(g, model, model, rep_size=args.representation_size, epoch=args.epochs,
                                        batch_size=1000, learning_rate=args.lr, negative_ratio=args.negative_ratio,
                                        ngmode=1, label_file=None, clf_ratio=args.clf_ratio, auto_save=True)
        model = trainer
    t2 = time.time()
    print(t2-t1)
    if args.method != 'gcn':
        print("Saving embeddings...")
        model.save_embeddings(args.output)
    if args.label_file and args.method != 'gcn':
        vectors = model.vectors
        X, Y = read_node_label(args.label_file)
        print("Training classifier using {:.2f}% nodes...".format(
            args.clf_ratio*100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, args.clf_ratio)

if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())
