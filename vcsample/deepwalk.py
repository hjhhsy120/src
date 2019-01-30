from __future__ import print_function
import time
import numpy as np
import random
from .graph import *

class deepwalk(object):
    # fac*node_size is the size of v_sampling table (for each epoch)
    def __init__(self, graph, fac=50, window=10):
        self.g = graph
        if graph.directed:
            self.g_r = None
        else:
            self.g_r = graph
        self.fac = fac
        self.window = window
        self.app = None
        self.it = None

    def sample_c(self, h):
        if self.it is None:
            self.it = {} # how many steps
            for i in self.g.G.nodes():
                self.it[i] = 0
            self.pl = {} # where it stopped
            self.neighbors = {}
            self.degrees = {}
            for root in self.g.G.nodes():
                self.neighbors[root] = list(self.g.G.neighbors(root))
                self.degrees[root] = len(self.neighbors[root])
        degrees = self.degrees
        neighbors = self.neighbors
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        i = 0
        batch_size = len(h)
        t = []
        while i < batch_size:
            root = look_back[h[i]]
            if degrees[root] == 0:
                t += [look_up[root]]
                i += 1
                continue
            if self.it[root] == 0:
                iid = root
                self.it[root] = self.window
            else:
                iid = self.pl[root]
            if degrees[iid] == 0:
                iid = root
                self.it[root] = self.window
            iid = random.choice(neighbors[iid])
            self.it[root] -= 1
            self.pl[root] = iid
            t += [look_up[iid]]
            i += 1
        return t

