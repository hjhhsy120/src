from __future__ import print_function
import time
import numpy as np
import random
from .graph import *

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


class deepwalk(object):
    # fac*node_size is the size of v_sampling table (for each epoch)
    def __init__(self, graph, max_iter=10, delta=0.01, fac=100, window=10, degree_bound=0, degree_power=1.0):
        self.g = graph
        if graph.directed:
            self.g_r = None
        else:
            self.g_r = graph
        self.window = window
        self.max_iter = max_iter
        self.fac = fac
        self.delta = delta
        self.sampling_table = None
        self.v = None
        self.alias_nodes = None
        self.direc = {}
        self.it = None

    def sample_v(self, batch_size):
        if self.sampling_table is None:
            self.page_rank(self.delta)
        sampling_table = self.sampling_table
        random.shuffle(sampling_table)
        i = 0
        while i < self.table_size - batch_size:
            yield sampling_table[i : i + batch_size]
            i += batch_size
        yield sampling_table[i:]

    def gen_alias(self):
        if self.v is None:
            self.page_rank(self.delta)
        v = self.v
        G = self.g.G
        degree = self.degree
        look_up = self.g.look_up_dict
        nodes = list(G.nodes())
        probs = {}
        nbr_r = {}
        alias_nodes = {}
        for node in nodes:
            probs[node] = {}
        for node in nodes:
            for nbr in nodes:
                probs[nbr][node] = v[look_up[node]] / degree[node] / v[look_up[nbr]]
        for node in nodes:
            s = sum(probs[node].values())
            nbr_r[node] = list(probs[node].keys())
            for i in nbr_r[node]:
                probs[node][i] /= s
            probs_v = list(probs[node].values())
            alias_nodes[node] = alias_setup(probs_v)
        self.nbr_r = nbr_r
        self.alias_nodes = alias_nodes

    def get_nxt(self, root, iid):
        if self.direc[root] == 0:
            cur_nbrs = list(self.g.G.neighbors(iid))
        else:
            cur_nbrs = self.nbr_r[root]
        if len(cur_nbrs) == 0:
            return None
        if self.direc[root] == 0:
            return random.choice(cur_nbrs)
        else:
            return cur_nbrs[alias_draw(self.alias_nodes[iid][0], self.alias_nodes[iid][1])]

    def sample_c(self, h):
        if self.alias_nodes is None:
            self.gen_alias()
        if self.it is None:
            self.it = {} # how many steps
            for i in self.g.G.nodes():
                self.it[i] = 0
                self.direc[i] = 1
            self.pl = {} # where it stopped
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        i = 0
        batch_size = len(h)
        t = []
        while i < batch_size:
            root = look_back[h[i]]
            if self.it[root] == 0:
                iid = root
                self.it[root] = self.window
                self.direc[root] = 1 - self.direc[root]
            else:
                iid = self.pl[root]
            iid = self.get_nxt(root, iid)
            if iid is None:
                iid = root
                self.it[root] = self.window
                self.direc[root] = 1 - self.direc[root]
                iid = self.get_nxt(root, iid)
            self.it[root] -= 1
            t += [look_up[iid]]
            i += 1
            while self.it[root] > 0:
                if i >= batch_size:
                    self.pl[root] = iid
                    break
                if h[i] != h[i - 1]:
                    self.pl[root] = iid
                    break
                self.it[root] -= 1
                iid = self.get_nxt(root, iid)
                if iid is None:
                    self.it[root] = 0
                    break
                t += [look_up[iid]]
                i += 1
        return t

    '''
    def sample_c(self, h):
        try:
            it = self.it
        except:
            self.it = {} # how many steps
            for i in self.g.G.nodes():
                self.it[i] = 0
            self.pl = {} # where it stopped
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        i = 0
        batch_size = len(h)
        t = []
        while i < batch_size:
            root = look_back[h[i]]
            if self.it[root] == 0:
                iid = root
                self.it[root] = self.window
            else:
                iid = self.pl[root]
            cur_nbrs = list(self.g.G.neighbors(iid))
            iid = random.choice(cur_nbrs)
            self.it[root] -= 1
            t += [look_up[iid]]
            i += 1
            while self.it[root] > 0:
                if i >= batch_size:
                    self.pl[root] = iid
                    break
                if h[i] != h[i - 1]:
                    self.pl[root] = iid
                    break
                self.it[root] -= 1
                cur_nbrs = list(self.g.G.neighbors(iid))
                iid = random.choice(cur_nbrs)
                t += [look_up[iid]]
                i += 1
        return t
    '''
    def page_rank(self, delta):
        cnt = 0
        nodes = []
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        node_size = self.g.G.number_of_nodes()
        nbrs = {}
        degree = {}
        v = np.zeros(node_size)
        v2 = np.zeros(node_size)
        for root in self.g.G.nodes():
            nbrs[root] = list(self.g.G.neighbors(root))
            degree[root] = len(nbrs[root])
            if degree[root] > 0:
                cnt += 1
                nodes += [root]
        self.degree = degree
        for i in nodes:
            v[look_up[i]] = 1 / cnt
        it = self.max_iter
        while True:
            if it == 0:
                break
            it -= 1
            for i in range(cnt):
                v2[i] = 0.0
                for j in nbrs[look_back[i]]:
                    v2[i] += v[look_up[j]] / degree[j]
            v2 = v2 / sum(v2)
            if sum(abs(v - v2)) <= delta:
                break
            v = v2
        self.v = v
        p = 0.0
        i = 0

        table_size = node_size * self.fac
        self.table_size = table_size
        self.sampling_table = np.zeros(table_size, dtype=np.uint32)
        for j in range(node_size):
            p += v[j]
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1

