
import sys
import numpy as np
import matplotlib.pyplot as plt

def cos_distance(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors

f = open('mydata/' + sys.argv[1] + '_degreeRank.txt', 'r')
degree_rank = {}
for l in f.readlines():
    ls = l.strip().split(' ')
    degree_rank[ls[0]] = int(ls[1])
f.close()
tot = len(degree_rank.keys())
vectors = load_embeddings(sys.argv[2])

ratios = [0.1, 0.2, 0.5, 0.6, 0.85, 0.9, 0.93, 0.95, 0.98, 1.0]
rank = {}
ranks = []
for x in ratios:
    idx = int(x * tot - 1)
    rank[idx] = x
    ranks += [idx]
for node in degree_rank.keys():
    if degree_rank[node] in ranks:
        arr = np.zeros((tot, tot))
        for n2 in degree_rank.keys():
            val = int(cos_distance(vectors[node], vectors[n2])*tot)
            for t in range(tot - val, tot):
                arr[t, degree_rank[n2]] = 1
        plt.matshow(arr, cmap='hot')
        plt.savefig('mypic3/' + sys.argv[1] + '_' + sys.argv[3] + '_' + str(rank[degree_rank[node]]) + '.png')
        plt.close()

