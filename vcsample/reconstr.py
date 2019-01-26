from __future__ import print_function
import numpy as np

def cos_distance(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

class reconstr(object):
    def __init__(self, g, vectors):
        nodes = list(g.G.nodes())
        tot = 0
        corr = 0
        cnt = 0
        for root in nodes:
            nbrs = list(g.G.neighbors(root))
            n_nbrs = len(nbrs)
            dis = {}
            for v in nodes:
                if root != v:
                    dis[v] = cos_distance(vectors[root], vectors[v])
            sorted_dis = sorted(dis.items(), key=lambda d: d[1], reverse=True)
            for v, x in sorted_dis[:n_nbrs]:
                if v in nbrs:
                    corr += 1
            tot += n_nbrs
        print("Graph reconstruction accuracy: {} ({} / {})".format(corr / tot, corr, tot))

