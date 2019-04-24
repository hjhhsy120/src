from __future__ import print_function
from vcgenerator import *

"""
jump_factor: prob to stop
Paper: Scalable Graph Embedding for Asymmetric Proximity
"""


class SimRank(VCGenerator):

    def __init__(self, graph, batch_size=1000, stop_factor=0.15, sample=400, step=10):
        super(SimRank, self).__init__()
        self.g = graph
        self.nodes = graph.G.nodes()
        self.stop_factor = stop_factor
        self.sample_per_node = sample
        self.step = step
        self._pairs_per_epoch = int(self.g.G.number_of_nodes() * self.sample_per_node/batch_size) * batch_size
        self.neighbors = {}
        self.degrees = {}
        for root in self.nodes():
            self.neighbors[root] = list(graph.G.neighbors(root))
            self.degrees[root] = len(self.neighbors[root])

    def pair_per_epoch(self):
        return self._pairs_per_epoch

    def generate_batch(self, batch_size):
        shuffle_nodes = list(self.nodes)
        look_up = self.g.look_up_dict
        random.shuffle(shuffle_nodes)
        h = []
        cnt = 0
        totcnt = 0
        while True:
            for root in shuffle_nodes:
                cnt += 1
                totcnt = totcnt + 1
                h += [look_up[root]]
                if cnt >= batch_size:
                    t = self.sample_context(h)
                    yield h, t
                    cnt = 0
                    h = []
            if totcnt >= self._pairs_per_epoch:
                break
        # if len(h) > 0:
        #     t = self.sample_context(h)
        #     yield h, t

    def sample_context(self, h):
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        t = []
        for i in h:
            root = look_back[i]
            iid = root
            s = self.step
            while s > 0:
                s -= 1
                stop = random.random()
                if stop < self.stop_factor:
                    break
                if self.degrees[iid] == 0:
                    break
                iid = random.choice(self.neighbors[iid])
            s = self.step
            while s > 0:
                s -= 1
                stop = random.random()
                if stop < self.stop_factor:
                    break
                if self.degrees[iid] == 0:
                    break
                iid = random.choice(self.neighbors[iid])
            t += [look_up[iid]]
        return t

