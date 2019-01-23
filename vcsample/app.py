from __future__ import print_function
import time
import numpy as np
import random

class APP(object):

    # jump_factor: prob to stop
    def __init__(self, graph, jump_factor=0.15, sample=50, step=10):
        self.g = graph
        self.jump_factor = jump_factor
        self.sample = sample
        self.step = step

    def sample_v(self, batch_size):
        random.seed()
        try:
            nodes = self.nodes
        except:
            nodes = []
            for root in self.g.G.nodes():
                nbrs = list(self.g.G.neighbors(root))
                if len(nbrs) > 0:
                    nodes += [root]
            self.nodes = nodes
        look_up = self.g.look_up_dict
        random.shuffle(nodes)
        h = []
        cnt = 0
        for root in nodes:
            for i in range(self.sample):
                cnt += 1
                h += [look_up[root]]
                if cnt >= batch_size:
                    yield h
                    cnt = 0
                    h = []

    def sample_c(self, h):
        jump_factor = self.jump_factor
        step = self.step
        G = self.g.G
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        t = []
        for i in h:
            root = look_back[i]
            cur_nbrs = list(G.neighbors(root))
            s = step - 1
            iid = random.choice(cur_nbrs)
            while s > 0:
                s -= 1
                jump = random.random()
                if jump < jump_factor:
                    break
                cur_nbrs = list(G.neighbors(iid))
                if len(cur_nbrs) == 0:
                    break
                iid = random.choice(cur_nbrs)
            t += [look_up[iid]]
        return t

    def batch_iter(self):
        random.seed()
        jump_factor = self.jump_factor
        sample = self.sample
        step = self.step
        batch_size = self.batch_size
        G = self.g.G
        look_up = self.g.look_up_dict
        nodes = list(G.nodes())
        random.shuffle(nodes)
        cnt = 0
        h = []
        t = []
        for root in nodes:
            cur_nbrs = list(G.neighbors(root))
            if len(cur_nbrs) == 0:
                continue
            for i in range(sample):
                s = step
                iid = -1
                while s > 0:
                    s -= 1
                    jump = random.random()
                    if jump < jump_factor:
                        break
                    iid = random.choice(cur_nbrs)
                    cur_nbrs = list(G.neighbors(iid))
                    if len(cur_nbrs) == 0:
                        break
                if iid != -1:
                    cnt += 1
                    h += [look_up[root]]
                    t += [look_up[iid]]
                    if cnt >= batch_size:
                        yield h, t, [1]
                        cnt = 0
                        h = []
                        t = []
                cur_nbrs = list(G.neighbors(root))
        if cnt > 0:
            yield h, t, [1]
