"""Graph utilities."""

# from time import time
import networkx as nx
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import random

__author__ = "Zhang Zhengyan"
__email__ = "zhangzhengyan14@mails.tsinghua.edu.cn"


class Graph(object):
    def __init__(self, prop_pos=0.5, prop_neg=0.5, prop_neg_tot=1.0):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0
        self.prop_pos = prop_pos
        self.prop_neg = prop_neg
        self.prop_neg_tot = prop_neg_tot

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_g(self, g):
        self.G = g
        self.encode_node()

    def read_adjlist(self, filename):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0
        self.encode_node()

    def read_edgelist(self, filename, weighted=False, directed=False):
        self.G = nx.DiGraph()
        self.directed = directed
        if directed:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)
        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()
        self.encode_node()

    def read_node_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()

    def read_node_features(self, filename):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array(
                [float(x) for x in vec[1:]])
        fin.close()

    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()

    def read_edge_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()

    def bfs(self, st, edges):
        q = {st}
        cnt = 1
        nbrs = {}
        look_up = self.look_up_dict
        for node in self.G.nodes():
            nbrs[node] = set()
        for e in edges:
            nbrs[e[0]].add(e[1])
            nbrs[e[1]].add(e[0])
        for node in self.G.nodes():
            nbrs[node] = list(nbrs[node])
            random.shuffle(nbrs[node])
        visited = np.zeros(self.node_size, dtype=np.uint32)
        visited[look_up[st]] = 1
        while cnt > 0:
            fst = q.pop()
            cnt -= 1
            for nxt in nbrs[fst]:
                x = look_up[nxt]
                if visited[x] == 0:
                    if (fst, nxt) in edges:
                        edges.remove((fst, nxt))
                    else:
                        edges.remove((nxt, fst))
                    if not self.directed:
                        edges.remove((nxt, fst))
                    visited[x] = 1
                    q.add(nxt)
                    cnt += 1


    def generate_pos_neg_links(self):
        """
        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.

        Modify graph by removing the postive links.
        """
        random.seed()
        # Select n edges at random (positive samples)
        look_up = self.look_up_dict
        nodes = list(self.G.nodes())
        n_nodes = self.node_size
        edges = set(self.G.edges())
        n_edges = len(edges)
        if not self.directed:
            n_edges = int(n_edges / 2)
        npos = int(self.prop_pos * n_edges)
        nneg = int(self.prop_neg_tot * n_edges)

        # if not nx.is_connected(self.G):
        #     raise RuntimeError("Input graph is not connected")

        # n_neighbors = [len(list(self.G.neighbors(v))) for v in nodes]
        # n_non_edges = n_nodes - 1 - np.array(n_neighbors)

        # non_edges = [e for e in nx.non_edges(self.G)]
        # print("Finding %d of %d non-edges" % (nneg, len(non_edges)))

        # Select m pairs of non-edges (negative samples)
        neg = set()
        neg_edge_list = []
        for i in range(nneg):
            while True:
                x1 = random.choice(nodes)
                x2 = random.choice(nodes)
                if not (x1, x2) in neg and not (x1, x2) in edges:
                    neg_edge_list += [(x1, x2)]
                    neg.add((x1, x2))
                    break


        print("Finding %d positive edges of %d total edges" % (npos, n_edges))

        # Find positive edges, and remove them.
        pos_edge_list = []
        n_count = 0

        st = random.choice(nodes)
        self.bfs(st, edges)

        edges = list(edges)
        random.shuffle(edges)
        for edge in edges:
            if self.directed or look_up[edge[0]] < look_up[edge[1]]:

                # Remove edge from graph
                self.G.remove_edge(*edge)
                if not self.directed:
                    self.G.remove_edge(edge[1], edge[0])

                pos_edge_list.append(edge)
                # print("Found: %d    " % (n_count), end="\r")
                n_count += 1

                # Exit if we've found npos nodes or we have gone through the whole list
                if n_count >= npos:
                    break

        if len(pos_edge_list) < npos:
            raise RuntimeWarning("Only %d positive edges found." % (n_count))

        self._pos_edge_list = pos_edge_list
        self._neg_edge_list = neg_edge_list

    def get_test_edges(self):
        nneg = len(self._neg_edge_list)
        edges = self._pos_edge_list + self._neg_edge_list[:int(self.prop_neg * nneg)]
        tot = len(edges)
        labels = np.zeros(tot)
        labels[:len(self._pos_edge_list)] = 1
        print('{} edges for test'.format(tot))
        return edges, labels

    def get_train_edges(self):
        if not self.directed:
            remaining_edges = []
            look_up = self.look_up_dict
            for edge in self.G.edges():
                if look_up[edge[0]] < look_up[edge[1]]:
                    remaining_edges += [edge]
        else:
            remaining_edges = list(self.G.edges())
        nneg = len(self._neg_edge_list)
        edges = remaining_edges + self._neg_edge_list[int(self.prop_neg * nneg):]
        tot = len(edges)
        labels = np.zeros(tot)
        labels[:len(remaining_edges)] = 1
        print('{} edges for training'.format(tot))
        return edges, labels
