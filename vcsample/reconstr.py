from __future__ import print_function
import numpy as np
from sklearn.neighbors import NearestNeighbors
'''
def cos_distance(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def reconstr(g, vectors, x):
    nodes = list(g.G.nodes())
    tot = 0
    corr = 0
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
'''
def reconstr(g, vectors, k_nbrs):
    nodes = list(vectors.keys())
    v = list(vectors.values())
    n_nodes = len(nodes)
    # normalize
    for i in range(n_nodes):
        v[i] = np.asarray(v[i]) / np.sqrt(sum(np.square(v[i])))
    neigh = NearestNeighbors(n_neighbors=k_nbrs)
    print("Computing KNN")
    neigh.fit(v)
    tot = 0
    corr = 0
    print("Getting neighbors")
    for i in range(n_nodes):
        root = nodes[i]
        nbrs = set(g.G.neighbors(root))
        n_nbrs = len(nbrs)
        if n_nbrs > 0:
            results = neigh.kneighbors([v[i]], n_nbrs, return_distance=False)
            for x in results[0]:
                if nodes[x] in nbrs:
                    corr += 1
            tot += n_nbrs
    print("Graph reconstruction accuracy: {} ({} / {})".format(corr / tot, corr, tot))
    print("{}\n".format(corr/tot))

