
import sys
import numpy as np
import matplotlib.pyplot as plt

maxcnt = int(sys.argv[1])
f = open('mydata/' + sys.argv[2] + '_degreeRank.txt', 'r')
degree_rank = {}
degrees = {}
for l in f.readlines():
    ls = l.strip().split(' ')
    degree_rank[ls[0]] = int(ls[1])
    degrees[ls[0]] = int(ls[2])
f.close()
tot = len(degree_rank.keys())
if tot > 5000:
    for i in degree_rank.keys():
        degree_rank[i] = int(degree_rank[i] * 5000 / tot)
    tot = 5000
f = open('aaa.txt', 'r')
arr = np.zeros((maxcnt, tot))
cnts = np.zeros(tot)
tt = 0
for l in f.readlines():
    ls = l.strip().split(' ')
    cnts[degree_rank[ls[0]]] += float(ls[2])
    if cnts[degree_rank[ls[0]]] >= maxcnt:
        cnts[degree_rank[ls[0]]] = maxcnt
    tt += 1
print (tt)
for i in degree_rank.keys():
    val = int(cnts[degree_rank[i]] / max(1, degrees[i]))
    for hh in range(max(maxcnt-val, 0), maxcnt):
        arr[hh, degree_rank[i]] = 1
plt.matshow(arr, cmap='hot')
plt.colorbar()
plt.savefig('mypic2/' + sys.argv[2] + '_' + sys.argv[3] + '_vdis.png')
