from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from .graph import *
from .classify import Classifier, read_node_label
import time

from . import app
from . import vctrainer
from . import deepwalk
from . import link
from .reconstr import reconstr
from .clustering import clustering

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

    parser.add_argument('--prop-pos', default=0.5, type=float,
                        help='proportion of positive edges for link prediction')
    parser.add_argument('--prop-neg', default=0.5, type=float,
                        help='proportion of negative edges for link prediction')
    parser.add_argument('--link-prediction', action='store_true',
                        help='Link prediction task.')
    parser.add_argument('--reconstruction', action='store_true',
                        help='Network reconstruction task.')
    parser.add_argument('--clustering', action='store_true',
                        help='Vertex clustering task.')
    parser.add_argument('--visualization', action='store_true',
                        help='Visualization after clustering task.')
    parser.add_argument('--exp-times', default=1, type=int,
                        help='How many times of experiments')
    args = parser.parse_args()

    if args.method != 'gcn' and not args.output:
        print("No output filename. Exit.")
        exit(1)

    return args

def main(args):
    if args.link_prediction:
        link.test_edge_functions(args)
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
        #model.save_embeddings(args.output)
    if args.reconstruction:
        reconstr(g, model.vectors)
        return

    if args.label_file and args.method != 'gcn':
        vectors = model.vectors
        labels = read_node_label(args.label_file)
        if args.clustering:
            r = clustering(model.vectors, labels, args)
            return
        X = list(labels.keys())
        Y = list(labels.values())
        print("Node classification test...")
        result = {}
        for ti in range(args.exp_times):
            clf = Classifier(vectors=vectors, clf=LogisticRegression())
            myresult = clf.split_train_evaluate(X, Y, args.clf_ratio)
            for nam in myresult.keys():
                if ti == 0:
                    result[nam] = myresult[nam]
                else:
                    result[nam] += myresult[nam]
        for nam in result.keys():
            print("{}:\t{}".format(nam, result[nam]/args.exp_times))

if __name__ == "__main__":
    random.seed()
    np.random.seed()
    main(parse_args())
