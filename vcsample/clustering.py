from __future__ import print_function
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from .visual import visual

class clustering(object):
    def __init__(self, vectors, labels, args):

        nodes = list(vectors.keys())
        v = list(vectors.values())
        all_labels = set()
        for l in labels.values():
            if len(l) != 1:
                print('Multi-label data is not suitable for clustering test!')
                exit()
            all_labels.add(l[0])
        K = len(all_labels)
        y_ = []
        for node in nodes:
            y_ += [labels[node][0]]
        a_nmi = 0.0
        exp_times = args.exp_times
        if args.visualization:
            exp_times = 1
        for i in range(exp_times):
            model = KMeans(n_clusters=K, init='k-means++')
            pred = model.fit_predict(v)
            nmi = normalized_mutual_info_score(y_, pred)
            print ('# {} NMI: {}'.format(i+1, nmi))
            a_nmi += nmi
            if args.visualization:
                visual(v, y_, args)
        print("Average NMI: {}".format(a_nmi / exp_times))



