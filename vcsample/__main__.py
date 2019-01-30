from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from .graph import *
from .classify import Classifier, read_node_label, load_embeddings
import time

from . import link
from .reconstr import reconstr
from .clustering import clustering, modularity
from .getmodel import getmodels, parse_args, emptymodel

# parse_args is moved to getmodel.py

def main(args):
    if args.link_prediction:
        print("Link prediction")
        link.test_edge_functions(args)
        return
    print("Reading...")
    g = Graph()
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted,
                    directed=args.directed)
    if not args.embedding_file:
        t1 = time.time()
        model = getmodels(g, args)
        t2 = time.time()
        print(t2-t1)
        if args.output:
            print("Saving embeddings...")
            model.save_embeddings(args.output)
    else:
        model = emptymodel(load_embeddings(args.embedding_file))

    if args.modularity:
        print("Modularity")
        modularity(g, model.vectors, args.min_k, args.max_k)

    if args.reconstruction:
        print("Graph reconstruction")
        reconstr(g, model.vectors, args.k_nbrs)

    if args.label_file:
        vectors = model.vectors
        labels = read_node_label(args.label_file)
        if args.clustering:
            print("Clustering")
            clustering(model.vectors, labels, args.exp_times)

        if args.classification:
            X = list(labels.keys())
            Y = list(labels.values())
            print("Node classification")
            clf_ratio_list = args.clf_ratio.strip().split(',')
            result_list = []
            for clf_ratio in clf_ratio_list:
                result = {}
                for ti in range(args.exp_times):
                    clf = Classifier(vectors=vectors, clf=LogisticRegression())
                    myresult = clf.split_train_evaluate(X, Y, float(clf_ratio))
                    for nam in myresult.keys():
                        if ti == 0:
                            result[nam] = myresult[nam]
                        else:
                            result[nam] += myresult[nam]
                for nam in result.keys():
                    print("clf_ratio = {}, {}: {}".format(clf_ratio, nam, result[nam]/args.exp_times))
                result_list += [result]
            exp_num = len(result_list)
            for i in range(exp_num):
                print("{}\t".format(clf_ratio_list[i]), end='')
            print("\nmicro")
            for i in range(exp_num):
                print("{}\t".format(result_list[i]["micro"]/args.exp_times), end='')
            print("\nmacro")
            for i in range(exp_num):
                print("{}\t".format(result_list[i]["macro"]/args.exp_times), end='')
            print("\n")


if __name__ == "__main__":
    random.seed()
    np.random.seed()
    main(parse_args())
