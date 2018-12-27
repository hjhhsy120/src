from __future__ import print_function
import time
from . import walker
from .trainer import trainer

def myparser0(sentences, window):
    mypairs = {}
    for se in sentences:
        l = len(se)
        for hi in range(l):
            for t in se[max(0, hi - window) : min(l, hi + window + 1)]:
                if se[hi] != t:
                    p = (se[hi], t)
                    if p in mypairs.keys():
                        mypairs[p] += 1.0
                    else:
                        mypairs[p] = 1.0
    return [{0: k[0], 1: k[1], "weight": mypairs[k]} for k in mypairs.keys()]

def myparser(sentences, window):
    samples = []
    for se in sentences:
        l = len(se)
        for hi in range(l):
            for t in se[max(0, hi - window) : min(l, hi + window + 1)]:
                if se[hi] != t:
                    samples += [{0: se[hi], 1: t, "weight": 1.0}]
    return samples

class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, dim, workers=8, p=1.0, q=1.0, dw=0, window=10,
            epoch=10, label_file=None, clf_ratio=0.5):

        if dw:
            p = 1.0
            q = 1.0

        self.graph = graph
        self.size = dim
        if dw == 1:
            self.walker = walker.BasicWalker(graph, workers=workers)
        elif dw == 2:
            self.walker = walker.MHWalker(graph, workers=workers)
        else:
            self.walker = walker.Walker(
                graph, p=p, q=q, workers=workers)
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
        sentences = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)

        samples = myparser(sentences, window)

        print("Learning representation...")
        self.model = trainer(graph=graph, samples=samples, rep_size=dim,
                            epoch=epoch, label_file=label_file, clf_ratio=clf_ratio,
                            batch_size=1000*window, negative_ratio=1, ran=False)
        self.vectors = self.model.vectors

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()


