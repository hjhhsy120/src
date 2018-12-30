from __future__ import print_function
import time
import numpy as np
import random
from .trainer import trainer

class APP(object):

    # jump_factor: prob to stop
    def __init__(self, graph, dim, jump_factor=0.15, iters=10, sample=50, step=10, negative_ratio=5,
                label_file=None, clf_ratio=0.5):
        random.seed()
        G = graph.G
        self.size = dim
        node_size = graph.node_size
        look_up = graph.look_up_dict
        # node_degree = np.zeros(node_size)  # out degree
        # for edge in G.edges():
        #     node_degree[look_up[edge[0]]
        #                 ] += G[edge[0]][edge[1]]["weight"]
        nodes = list(G.nodes())
        print('Walking...')
        samples = []
        for kk in range(iters):
            random.shuffle(nodes)
            for root in nodes:
                cur_nbrs = list(G.neighbors(root))
                if len(cur_nbrs) == 0:
                    continue
                for i in range(sample+1):
                    s = step
                    iid = -1
                    while s > 0:
                        s -= 1
                        jump = random.random()
                        if jump < jump_factor:
                            break
                        iid = random.choice(cur_nbrs)
                        cur_nbrs = list(G.neighbors(iid))
                    if iid != -1:
                        samples.append({0: root, 1: iid, "weight": 1.0})
                    cur_nbrs = list(G.neighbors(root))
        print('Training...')
        self.model = trainer(graph, samples, rep_size=dim, batch_size=1000, epoch=10,
                    negative_ratio=negative_ratio, label_file=label_file, clf_ratio=clf_ratio,
                    ran=False, ngmode=1)
        self.vectors = self.model.vectors

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
