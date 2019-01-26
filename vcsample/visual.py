import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

class visual(object):
    def __init__(self, vectors_, labels_, args):

        tsne = manifold.TSNE(n_components=2, init='pca')
        X_tsne = tsne.fit_transform(vectors_)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # πÈ“ªªØ
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(labels_[i]), color=plt.cm.Set1(labels_[i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()




