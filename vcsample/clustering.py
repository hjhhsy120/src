from __future__ import print_function
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

def modularity(g, vectors, min_k, max_k):
    nodes = list(vectors.keys())
    v = list(vectors.values())
    n_nodes = len(nodes)
    best_m = -1
    best_k = 0
    degree = {}
    for K in range(min_k, max_k + 1):
        model = KMeans(n_clusters=K, init='k-means++')
        pred = model.fit_predict(v)
        community_set = set()
        community = {}
        nodes_in_c = {}
        for i in range(n_nodes):
            c = pred[i]
            node = nodes[i]
            community[node] = c
            if c in community_set:
                nodes_in_c[c] += [node]
            else:
                community_set.add(c)
                nodes_in_c[c] = [node]
        edge_inside = {}
        degree_sum = {}
        for c in community_set:
            edge_inside[c] = 0
            degree_sum[c] = 0
            for node in nodes_in_c[c]:
                degree_sum[c] += g.G.degree[node]
            if not g.directed:
                degree_sum[c] /= 2
        edges = list(g.G.edges())
        m = len(edges)
        for edge in edges:
            c = community[edge[0]]
            if c == community[edge[1]]:
                edge_inside[c] += 1
        if g.directed:
            for c in community_set:
                edge_inside[c] *= 2
        else:
            m /= 2
        result = 0.0
        for c in community_set:
            result += edge_inside[c] / (2 * m) - (degree_sum[c]/(2*m))*(degree_sum[c]*1.0/(2*m))
        if result > best_m:
            best_m = result
            best_k = K
        print("K = {}; modularity = {}".format(K, result))
    print("best K: {}".format(best_k))
    print("best modularity: {}".format(best_m))
    print("{}\t{}\n".format(best_k, best_m))

def clustering(vectors, labels, exp_times):
    nodes = list(vectors.keys())
    v = list(vectors.values())
    all_labels = set()
    for l in labels.values():
        if len(l) != 1:
            print('Multi-label data is not suitable for clustering test!')
            return
        all_labels.add(l[0])
    K = len(all_labels)
    y_ = []
    for node in nodes:
        y_ += [labels[node][0]]
    a_nmi = 0.0

    for i in range(exp_times):
        model = KMeans(n_clusters=K, init='k-means++')
        pred = model.fit_predict(v)
        nmi = normalized_mutual_info_score(y_, pred)
        print ('# {} NMI: {}'.format(i+1, nmi))
        a_nmi += nmi
    print("Average NMI: {}\n".format(a_nmi / exp_times))
