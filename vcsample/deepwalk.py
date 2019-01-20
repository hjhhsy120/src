from __future__ import print_function
import time
import numpy as np
import random

class deepwalk(object):
    # fac*node_size is the size of v_sampling table (for each epoch)
    def __init__(self, graph, window=10, walk_length=10, fac=100):
        self.g = graph
        self.window = window
        self.walk_length = walk_length
        self.fac = fac

    def sample_v(self, batch_size):
        try:
            sampling_table = self.sampling_table
        except:
            sampling_table = self.page_rank(0.01)
        random.shuffle(sampling_table)
        i = 0
        while i < self.table_size - batch_size:
            yield sampling_table[i : i + batch_size]
            i += batch_size
        yield sampling_table[i:]

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

    def page_rank(self, delta):
        cnt = 0
        nodes = []
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        node_size = self.g.G.number_of_nodes()
        table_size = node_size * self.fac
        self.table_size = table_size
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
        for i in nodes:
            v[look_up[i]] = 1 / cnt
        it = self.walk_length
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
        p = 0.0
        i = 0
        self.sampling_table = np.zeros(table_size, dtype=np.uint32)
        for j in range(node_size):
            p += v[j]
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1
        return self.sampling_table

