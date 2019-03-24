from __future__ import print_function
import math
from graph import *
from vcgenerator import *
import walker

class Node2vec(VCGenerator):
    def __init__(self, graph, fac=50, window=10, degree_bound=0, degree_power=1.0, p=1.0, q=1.0):
        super(Node2vec, self).__init__()
        self.g = graph
        self.nodes = graph.G.nodes()
        self.node_size = len(self.nodes)
        if graph.directed:
            self.g_r = None
        else:
            self.g_r = graph
        self.fac = fac
        self.window = window
        self.degree_bound = degree_bound
        self.degree_power = degree_power
        self.p = p
        self.q = q

        self.neighbors = {}
        self.degrees = {}
        for root in self.g.G.nodes():
            self.neighbors[root] = list(self.g.G.neighbors(root))
            self.degrees[root] = len(self.neighbors[root])

        self.build_node_tables()

        # variable stores temporal information of random walks
        self.pl = {}
        self.walks = {}
        self.walkslen = {}
        for i in self.g.G.nodes():
            self.pl[i] = self.window
            self.walkslen[i] = self.window

        self.walker = None

    def build_node_tables(self):
        table_size = self.fac * self.node_size

        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = np.zeros(self.node_size)

        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            node_degree[look_up[edge[0]]] += self.g.G[edge[0]][edge[1]]["weight"]

        degree_bound = self.degree_bound
        if degree_bound > 0:
            for i in range(self.node_size):
                if node_degree[i] > degree_bound:
                    node_degree[i] = degree_bound

        for i in range(self.node_size):
            node_degree[i] = math.pow(node_degree[i], self.degree_power)

        norm = sum([node_degree[i] for i in range(self.node_size)])

        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(self.node_size):
            p += float(node_degree[j]) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1

    def generate_batch(self, batch_size):
        table_size = len(self.sampling_table)
        random.shuffle(self.sampling_table)
        i = 0
        while i < table_size:
            j = min(i + batch_size, table_size)
            h = self.sampling_table[i:j]
            t = self.sample_context(h)
            yield h, t
            i = j

    def sample_context(self, h):
        # return self.sample_c_asym(h)
        return self.sample_c_asym(h)

    def sample_c_asym(self, h):
        if self.walker is None:
            self.walker = walker.Walker(
                self.g, p=self.p, q=self.q, workers=1)
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()

        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list
        i = 0
        batch_size = len(h)
        t = []
        while i < batch_size:
            root = look_back[h[i]]

            if self.degrees[root] == 0:
                t += [look_up[root]]
                i += 1
                continue
            if self.pl[root] == self.walkslen[root]:
                self.walks[root] = self.walker.node2vec_walk(
                    walk_length=self.window, start_node=root)
                self.pl[root] = 1
                self.walkslen[root] = len(self.walks[root])
            t += [look_up[self.walks[root][self.pl[root]]]]
            self.pl[root] += 1
            i += 1
        return t

    # def sample_c_sym(self, h):
    #
    #     degrees = self.degrees
    #     neighbors = self.neighbors
    #     look_up = self.g.look_up_dict
    #     look_back = self.g.look_back_list
    #     i = 0
    #     batch_size = len(h)
    #     new_h = []
    #     t = []
    #     while i < batch_size:
    #         try:
    #             root = look_back[h[i]]
    #         except:
    #             print(h[i])
    #             exit()
    #         if degrees[root] == 0:
    #             t += [look_up[root]]
    #             i += 1
    #             continue
    #         if self.it[root] == 0:
    #             iid = root
    #             self.it[root] = self.window
    #         else:
    #             iid = self.pl[root]
    #         if degrees[iid] == 0:
    #             iid = root
    #             self.it[root] = self.window
    #         iid = random.choice(neighbors[iid])
    #         self.it[root] -= 1
    #         self.pl[root] = iid
    #
    #         new_h += [h[i]]
    #         t += [look_up[iid]]
    #
    #         new_h += [look_up[iid]]
    #         t += [h[i]]
    #         i += 1
    #     return new_h, t

    # def sample_batch(self, batch_size, negative_ratio):
    #     try:
    #         nodes = self.nodes
    #     except:
    #         self.nodes = list(self.g.G.nodes())
    #         nodes = self.nodes
    #     numNodes = len(nodes)
    #     table_size = self.fac * numNodes
    #
    #     print("Pre-procesing for non-uniform negative sampling!")
    #     node_degree = np.zeros(numNodes)
    #
    #     look_up = self.g.look_up_dict
    #     for edge in self.g.G.edges():
    #         node_degree[look_up[edge[0]]
    #                     ] += self.g.G[edge[0]][edge[1]]["weight"]
    #     degree_bound = self.degree_bound
    #     if degree_bound > 0:
    #         for i in range(numNodes):
    #             if node_degree[i] > degree_bound:
    #                 node_degree[i] = degree_bound
    #
    #     for i in range(numNodes):
    #         node_degree[i] = math.pow(node_degree[i], self.degree_power)
    #
    #     norm = sum([node_degree[i] for i in range(numNodes)])
    #
    #     sampling_table = np.zeros(int(table_size), dtype=np.uint32)
    #
    #     p = 0
    #     i = 0
    #     for j in range(numNodes):
    #         p += float(node_degree[j]) / norm
    #         while i < table_size and float(i) / table_size < p:
    #             sampling_table[i] = j
    #             i += 1
    #     random.shuffle(sampling_table)
    #     i = 0
    #     # TODO: full batch training
    #     # batch_size = table_size
    #     while i < table_size:
    #         h = []
    #         t = []
    #         sign = []
    #         j = min(i + batch_size, table_size)
    #         hx = sampling_table[i:j]
    #         tx = self.sample_c_asym(hx)
    #         for idx in range(len(hx)):
    #             h.append(hx[idx])
    #             t.append(tx[idx])
    #             sign.append(1.0)
    #             for negId in range(negative_ratio):
    #                 h.append(hx[idx])
    #                 t.append(random.randint(0, numNodes -1))
    #                 sign.append(0)
    #         yield h, t, sign
    #         i = j

    # def sample_v(self, batch_size):
    #     try:
    #         nodes = self.nodes
    #     except:
    #         self.nodes = list(self.g.G.nodes())
    #         nodes = self.nodes
    #     numNodes = len(nodes)
    #     table_size = self.fac * numNodes
    #
    #     print("Pre-procesing for non-uniform negative sampling!")
    #     node_degree = np.zeros(numNodes)
    #
    #     look_up = self.g.look_up_dict
    #     for edge in self.g.G.edges():
    #         node_degree[look_up[edge[0]]
    #                     ] += self.g.G[edge[0]][edge[1]]["weight"]
    #     degree_bound = self.degree_bound
    #     if degree_bound > 0:
    #         for i in range(numNodes):
    #             if node_degree[i] > degree_bound:
    #                 node_degree[i] = degree_bound
    #
    #     for i in range(numNodes):
    #         node_degree[i] = math.pow(node_degree[i], self.degree_power)
    #
    #     norm = sum([node_degree[i] for i in range(numNodes)])
    #
    #     sampling_table = np.zeros(int(table_size), dtype=np.uint32)
    #
    #     p = 0
    #     i = 0
    #     h = []
    #     for j in range(numNodes):
    #         p += float(node_degree[j]) / norm
    #         while i < table_size and float(i) / table_size < p:
    #             sampling_table[i] = j
    #             i += 1
    #     random.shuffle(sampling_table)
    #     i = 0
    #     while i < table_size:
    #         j = min(i + batch_size, table_size)
    #         yield sampling_table[i:j]
    #         i = j

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
        t = self.sample_c_asym(h)
        for idx in range(len(h)):
            fout.write("{} {}\n".format(h[idx],t[idx]))
        fout.close()
