from __future__ import print_function
import math
from graph import *
from vcgenerator import *
import numpy as np
import scipy
from scipy import sparse
# import networkx.convert_matrix.to_scipy_sparse_matrix

# notice: this is different from alias_setup in openne!
# probs is a list of tuples (index, probability) here instead of a list of probability
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
        q[kk] = K*prob[1]
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

class SimRank(VCGenerator):
    # damp is C in the formula
    def __init__(self, graph, fac=50, maxIteration=10, damp=0.8):
        super(SimRank, self).__init__()
        self.g = graph
        self.nodes = list(graph.G.nodes())
        self.node_size = len(self.nodes)
        self.look_back_list = graph.look_back_list
        self.fac = fac
        self.simranktable = None
        self.maxIteration = maxIteration
        self.damp = damp

    def gentable(self):
        node_size = self.node_size
        look_up = self.g.look_up_dict
        damp = self.damp
        G = self.g.G
        xs = []
        ys = []
        data = []
        for node in self.nodes:
            neighbors = list(G.neighbors(node))
            num = len(neighbors)
            summ = 0
            for j in neighbors:
                summ += G[node][j]["weight"]
            xs += [look_up[node]] * num
            ys += [look_up[j] for j in neighbors]
            data += [G[node][j]["weight"] / summ for j in neighbors]

        # trans_matrix = to_scipy_sparse_matrix(self.g.G,
        #             [self.look_back_list[i] for i in range(node_size)], dtype=np.float32)
        trans_matrix = sparse.csr_matrix((data, (xs, ys)), dtype=np.float32, shape=(node_size, node_size))
        ids = [i for i in range(node_size)]
        data = [1 - damp for i in range(node_size)]
        sim_matrix = sparse.csr_matrix((data, (ids, ids)), dtype=np.float32, shape=(node_size, node_size))
        ini = sparse.csr_matrix((data, (ids, ids)), dtype=np.float32, shape=(node_size, node_size))
        for i in range(self.maxIteration):
            print ("simrank iteration %d:" % (i + 1))
            sim_matrix = damp * trans_matrix.transpose().dot(
                    sim_matrix).dot(trans_matrix) + ini
        simranktable = {}
        probs = {}
        for i in range(node_size):
            sim = sim_matrix.getrow(i)
            nz = sim.nonzero()
            summ = 0.0
            data = sim.data
            data /= sum(data)
            tt = len(data)
            probs[i] = []
            for j in range(tt):
                probs[i] += [(nz[1][j], data[j])]
            simranktable[i] = alias_setup(probs[i])
        self.simranktable = simranktable
        self.probs = probs

    def generate_batch(self, batch_size):
        shuffle_nodes = list(self.nodes)
        look_up = self.g.look_up_dict
        random.shuffle(shuffle_nodes)
        h = []
        cnt = 0
        for i in range(self.fac):
            for root in shuffle_nodes:
                cnt += 1
                h += [look_up[root]]
                if cnt >= batch_size:
                    t = self.sample_context(h)
                    yield h, t
                    cnt = 0
                    h = []
        if len(h) > 0:
            t = self.sample_context(h)
            yield h, t

    def sample_context(self, h):
        if self.simranktable is None:
            self.gentable()
        return [self.probs[node][alias_draw(self.simranktable[node][0], self.simranktable[node][1])][0] for node in h]

    def gendata(self, output):
        fout = open(output, 'w')

        try:
            nodes = self.nodes
        except:
            self.nodes = list(self.g.G.nodes())
            nodes = self.nodes
        numNodes = len(nodes)
        table_size = self.fac * numNodes

        node_degree = np.zeros(numNodes)

        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            node_degree[look_up[edge[0]]
                        ] += self.g.G[edge[0]][edge[1]]["weight"]
        degree_bound = self.degree_bound
        if degree_bound > 0:
            for i in range(numNodes):
                if node_degree[i] > degree_bound:
                    node_degree[i] = degree_bound

        for i in range(numNodes):
            node_degree[i] = math.pow(node_degree[i], self.degree_power)

        norm = sum([node_degree[i] for i in range(numNodes)])

        sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        h = []
        for j in range(numNodes):
            p += float(node_degree[j]) / norm
            while i < table_size and float(i) / table_size < p:
                sampling_table[i] = j
                i += 1
        random.shuffle(sampling_table)

        h = sampling_table
        t = self.sample_context(h)
        for idx in range(len(h)):
            fout.write("{} {}\n".format(h[idx],t[idx]))
        fout.close()
