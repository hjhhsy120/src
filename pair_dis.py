
import sys
import numpy as np
import matplotlib.pyplot as plt

f = open('mydata/' + sys.argv[1] + '_degreeRank.txt', 'r')
degree_rank = {}
for l in f.readlines():
    ls = l.strip().split(' ')
    degree_rank[ls[0]] = int(ls[1])
f.close()

nodes = list(degree_rank.keys())
tot = len(nodes)
arr = np.zeros((tot, tot))

f = open('aaa.txt', 'r')
for l in f.readlines():
    ls = l.strip().split(' ')
    arr[degree_rank[ls[0]], degree_rank[ls[1]]] += float(ls[2])

ratios = [0.1, 0.2, 0.5, 0.6, 0.85, 0.9, 0.93, 0.95, 0.98, 1.0]
rank = {}
ranks = []
for x in ratios:
    idx = int(x * tot - 1)
    rank[idx] = x
    ranks += [idx]
for node in nodes:
    if degree_rank[node] in ranks:
        dislist = []
        i = degree_rank[node]
        for j in range(tot):
            # val = int(cos_distance(vectors[node], vectors[n2])*tot)
            dislist += [arr[i,j]]
        dislist.sort(reverse=True)
        x = [i for i in range(tot)]
        # plt.ylim((0.0, 1.0))
        plt.scatter(x, dislist, s=1)
        plt.savefig('mypic6/' + sys.argv[1] + '_' + sys.argv[2] + '_' + str(rank[degree_rank[node]]) + '_' + node + '.png')
        plt.close()

