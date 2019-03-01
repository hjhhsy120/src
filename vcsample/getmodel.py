from __future__ import print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from . import vctrainer
from . import app
from . import deepwalk
from . import combine

class emptymodel(object):
    def __init__(self, vectors):
        self.vectors = vectors

def parse_args():

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    # input files
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output',
                        help='Output representation file')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    # parser.add_argument('--feature-file', default='',
    #                    help='The file of node features')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--embedding-file',
                        help='Pretrained embedding file')

    # embedding training parameters
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='The training epochs')
    parser.add_argument('--epoch-fac', default=50, type=int,
                        help='epoch-fac * node num in graph = node num per epoch')
    parser.add_argument('--batch-size', default=1000, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='the negative ratio of embedding training')

    # algorithm parameters
    parser.add_argument('--model-v', required=True,
                        help='The vertex sampling model')
    parser.add_argument('--model-c',
                        help='The context sampling model')

    # APP
    parser.add_argument('--app-jump-factor', default=0.15, type=float,
                        help='Jump factor (APP)')
    parser.add_argument('--app-step', default=80, type=int,
                        help='Maximum number of walking steps(APP)')

    # deepwalk
    parser.add_argument('--degree-power', default=1.0, type=float,
                        help='Bound of degree for sample_v of deepwalk.')
    parser.add_argument('--degree-bound', default=0, type=int,
                        help='Bound of degree for sample_v of deepwalk.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')

    # combination
    parser.add_argument('--combine', default=0.5, type=float,
                        help='Combine A and B with how much A.')

    # parser.add_argument('--p', default=1.0, type=float)
    # parser.add_argument('--q', default=1.0, type=float)

    # parser.add_argument('--dropout', default=0.5, type=float,
    #                    help='Dropout rate (1 - keep probability)')
    # parser.add_argument('--weight-decay', type=float, default=5e-4,
    #                    help='Weight for L2 loss on embedding matrix')
    # parser.add_argument('--hidden', default=16, type=int,
    #                    help='Number of units in hidden layer 1')
    #parser.add_argument('--kstep', default=4, type=int,
    #                    help='Use k-step transition probability matrix')
    #parser.add_argument('--lamb', default=0.2, type=float,
    #                    help='lambda is a hyperparameter in TADW')

    #parser.add_argument('--alpha', default=1e-6, type=float,
    #                    help='alhpa is a hyperparameter in SDNE')
    #parser.add_argument('--beta', default=5., type=float,
    #                    help='beta is a hyperparameter in SDNE')
    #parser.add_argument('--nu1', default=1e-5, type=float,
    #                    help='nu1 is a hyperparameter in SDNE')
    #parser.add_argument('--nu2', default=1e-4, type=float,
    #                    help='nu2 is a hyperparameter in SDNE')
    #parser.add_argument('--bs', default=200, type=int,
    #                    help='batch size of SDNE')
    #parser.add_argument('--encoder-list', default='[1000, 128]', type=str,
    #                    help='a list of numbers of the neuron at each encoder layer, the last number is the '
    #;                         'dimension of the output node representation')

    # evaluation parameters
    parser.add_argument('--exp-times', default=10, type=int,
                        help='How many times of experiments')

    parser.add_argument('--classification', action='store_true',
                        help='Node classification task.')
    parser.add_argument('--clf-ratio', default="0.5",
                        help='The list for ratio of training data in the classification, separated by ,')

    parser.add_argument('--link-prediction', action='store_true',
                        help='Link prediction task.')
    parser.add_argument('--prop-pos', default=0.5, type=float,
                        help='proportion of positive edges for link prediction')
    parser.add_argument('--prop-neg', default=0.5, type=float,
                        help='proportion of negative edges for link prediction')
    parser.add_argument('--prop-neg-tot', default=1.0, type=float,
                        help='total proportion of negative edges for link prediction')
    parser.add_argument('--cached-fn', default='',
                        help='name of cached/to-be-cached graph file for link prediction task.')

    parser.add_argument('--reconstruction', action='store_true',
                        help='Network reconstruction task.')
    parser.add_argument('--k-nbrs', default=30, type=int,
                        help='K for knn in reconstruction')

    parser.add_argument('--clustering', action='store_true',
                        help='Vertex clustering task testing NMI.')
    parser.add_argument('--modularity', action='store_true',
                        help='Vertex clustering task testing modularity')
    parser.add_argument('--min-k', default=2, type=int,
                        help='minimum k for modularity')
    parser.add_argument('--max-k', default=30, type=int,
                        help='maximum k for modularity')
    args = parser.parse_args()

    return args

def getmodel(model, g, args):
    if model == 'deepwalk':
        return deepwalk.deepwalk(graph=g, fac=args.epoch_fac, window=args.window_size,
                                degree_bound=args.degree_bound, degree_power=args.degree_power)
    if model == 'app':
        return app.APP(graph=g, jump_factor=args.app_jump_factor, sample=args.epoch_fac, step=args.app_step)

    if model == 'deepwalk,app':
        return combine.combine(g, args)

    model_list = ['app', 'deepwalk', 'deepwalk,app']
    print ("The sampling method does not exist!")
    print ("Please choose from the following:")
    for m in model_list:
        print(m)
    exit()

def getmodels(g, args):

    model_v = getmodel(args.model_v, g, args)
    if not args.model_c:
        model_c = model_v
    elif args.model_c == args.model_v:
        model_c = model_v
    else:
        model_c = getmodel(args.model_c, g, args)

    trainer = vctrainer.vctrainer(g, model_v, model_c, rep_size=args.representation_size,
                                        epoch=args.epochs, batch_size=args.batch_size,
                                        learning_rate=args.lr, negative_ratio=args.negative_ratio)
    return trainer
