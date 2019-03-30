from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from graph import *
import time
from getmodel import getmodels

def parse_args():

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    # input files
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output', required=True,
                        help='Output embedding file')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')

    # embedding training parameters
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='The training epochs')
    parser.add_argument('--epoch-fac', default=50, type=int,
                        help='epoch-fac * node num in graph = node num per epoch')
    parser.add_argument('--batch-size', default=1000, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.2, type=float,
                        help='learning rate')  # TODO: learning rate
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='the negative ratio of embedding training')

    # algorithm parameters
    parser.add_argument('--model-v', required=True,
                        help='The vertex sampling model')
    parser.add_argument('--model-c',
                        help='The context sampling model')
    parser.add_argument('--emb_model',
                        help='The embedding learning model')
    parser.add_argument('--rw-file',
                        help='The file path of random walk')

    # APP
    parser.add_argument('--app-jump-factor', default=0.15, type=float,
                        help='Jump factor (APP)')
    parser.add_argument('--app-sample', default=200, type=int,
                        help='Jump factor (APP)')
    parser.add_argument('--app-step', default=10, type=int,
                        help='Maximum number of walking steps(APP)')

    # deepwalk & node2vec
    parser.add_argument('--degree-power', default=1.0, type=float,
                        help='Bound of degree for sample_v of deepwalk.')
    parser.add_argument('--degree-bound', default=0, type=int,
                        help='Bound of degree for sample_v of deepwalk.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')

    # node2vec
    parser.add_argument('--node2vec-p', default=1.0, type=float,
                        help='Value p for node2vec.')
    parser.add_argument('--node2vec-q', default=1.0, type=float,
                        help='Value q for node2vec.')

    # fixedpair
    parser.add_argument('--pair-file',
                        help='file path of pairs')

    # combination
    parser.add_argument('--combine', default=0.5, type=float,
                        help='Combine A and B with how much A.')

    # simrank
    parser.add_argument('--simrank-maxiter', default=10, type=int,
                        help='Max iterations for simrank.')
    parser.add_argument('--simrank-damp', default=0.8, type=float,
                        help='Value damp for simrank.')

    ##multi-threading
    parser.add_argument('--thread-num', default=4, type=int,
                        help='Number of threads.')

    args = parser.parse_args()

    return args

def print_args(args):
    print("==================")
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    print("==================")

def main(args):
    print_args(args)

    print("Reading Graph ...")
    g = Graph()
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed)

    t1 = time.time()
    model = getmodels(g, args)
    t2 = time.time()
    print("Train time cost: {}".format(t2-t1))
    print("Saving embeddings...")
    model.save_embeddings(args.output)

if __name__ == "__main__":
    random.seed()
    np.random.seed()
    main(parse_args())
