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
    def __init__(self, prop_pos=0.5, prop_neg=0.5):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0
        self.prop_pos = prop_pos
        self.prop_neg = prop_neg

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
            self.uedges = []
            def read_unweighted(l):
                src, dst = l.split()
                if src == dst:
                    return
                if not (dst, src) in self.uedges:
                    self.uedges += [(src, dst)]
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                if src == dst:
                    return
                if not (dst, src) in self.uedges:
                    self.uedges += [(src, dst)]
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

    def generate_pos_neg_links(self):
        """
        Select random existing edges in the graph to be postive links,
        and random non-edges to be negative links.

        Modify graph by removing the postive links.
        """
        random.seed()
        # Select n edges at random (positive samples)
        nodes = list(self.G.nodes())
        n_nodes = len(nodes)
        all_edges = list(self.G.edges())
        if self.directed:
            edges = all_edges
        else:
            edges = self.uedges
        n_edges = len(edges)
        npos = int(self.prop_pos * n_edges)
        nneg = int(self.prop_neg * n_edges)

        # if not nx.is_connected(self.G):
        #     raise RuntimeError("Input graph is not connected")

        # n_neighbors = [len(list(self.G.neighbors(v))) for v in nodes]
        # n_non_edges = n_nodes - 1 - np.array(n_neighbors)

        # non_edges = [e for e in nx.non_edges(self.G)]
        # print("Finding %d of %d non-edges" % (nneg, len(non_edges)))

        # Select m pairs of non-edges (negative samples)
        neg_edge_list = []
        for i in range(nneg):
            while True:
                x1 = random.choice(nodes)
                x2 = random.choice(nodes)
                if not (x1, x2) in neg_edge_list and not (x2, x1) in neg_edge_list and not (x1, x2) in all_edges:
                    neg_edge_list += [(x1, x2)]
                    break


        print("Finding %d positive edges of %d total edges" % (npos, n_edges))

        # Find positive edges, and remove them.
        pos_edge_list = []
        n_count = 0
        n_ignored_count = 0
        random.shuffle(edges)
        for edge in edges:

            # Remove edge from graph
            data = self.G[edge[0]][edge[1]]
            self.G.remove_edge(*edge)
            if not self.directed:
                data_r = self.G[edge[1]][edge[0]]
                self.G.remove_edge(edge[1], edge[0])

            # Check if graph is still connected
            #TODO: We shouldn't be using a private function for bfs
            reachable_from_v1 = nx.connected._plain_bfs(self.G, edge[0])
            if edge[1] not in reachable_from_v1:
                self.G.add_edge(*edge, **data)
                if not self.directed:
                    self.G.add_edge(edge[1], edge[0], **data_r)
                n_ignored_count += 1
            else:
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

    def get_selected_edges(self):
        edges = self._pos_edge_list + self._neg_edge_list
        labels = np.zeros(len(edges))
        labels[:len(self._pos_edge_list)] = 1
        return edges, labels
