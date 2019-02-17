
import sys
import numpy as np
import matplotlib.pyplot as plt

maxcnt = int(sys.argv[1])
f = open('mydata/' + sys.argv[2] + '_degreeRank.txt', 'r')
degree_rank = {}
for l in f.readlines():
    ls = l.strip().split(' ')
    degree_rank[ls[0]] = int(ls[1])
f.close()
tot = len(degree_rank.keys())
if tot > 5000:
    for i in degree_rank.keys():
        degree_rank[i] = int(degree_rank[i] * 5000 / tot)
    tot = 5000
f = open('aaa.txt', 'r')
arr = np.zeros((tot, tot))
tt = 0
for l in f.readlines():
    ls = l.strip().split(' ')
    arr[degree_rank[ls[0]], degree_rank[ls[1]]] += float(ls[2])
    if arr[degree_rank[ls[0]], degree_rank[ls[1]]] >= maxcnt:
        arr[degree_rank[ls[0]], degree_rank[ls[1]]] = maxcnt
    tt += 1
print (tt)
plt.matshow(arr, cmap='hot')
plt.colorbar()
plt.savefig('mypic2/' + sys.argv[2] + '_' + sys.argv[3] + '.png')
