from __future__ import print_function
from .trainer import trainer

class LINE(object):

    def __init__(self, graph, rep_size=128, batch_size=1000, epoch=10, negative_ratio=5, label_file=None, clf_ratio=0.5, auto_save=True):
        self.g = graph
        self.rep_size = rep_size
        samples = [{0: edge[0], 1: edge[1], "weight": self.g.G[edge[0]][edge[1]]["weight"]}
                    for edge in self.g.G.edges()]

        self.model = trainer(graph, samples, rep_size, batch_size, epoch, negative_ratio, label_file, clf_ratio, auto_save)
        self.vectors = self.model.vectors

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()

