

import sys

clf_ratio = 0.5
turnnum = 5
model_name = sys.argv[1]
data_name = sys.argv[2]

import numpy
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from time import time
from sklearn.linear_model import LogisticRegression

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return numpy.asarray(all_labels)


class Classifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        print(results)
        return results

    def predict(self, X, top_k_list):
        X_ = numpy.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, D):
        state = numpy.random.get_state()

        training_size = int(train_precent * len(X))
        numpy.random.seed()
        shuffle_indices = numpy.random.permutation(numpy.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
        tot = len(X)
        X_020 = []
        Y_020 = []
        X_2080 = []
        Y_2080 = []
        X_80100 = []
        Y_80100 = []
        for i in range(training_size, tot):
            j = shuffle_indices[i]
            if D[X[j]] <= tot * 0.2:
                X_020 += [X[j]]
                Y_020 += [Y[j]]
            elif D[X[j]] <= tot * 0.8:
                X_2080 += [X[j]]
                Y_2080 += [Y[j]]
            else:
                X_80100 += [X[j]]
                Y_80100 += [Y[j]]

        self.train(X_train, Y_train, Y)
        numpy.random.set_state(state)
        return [self.evaluate(X_test, Y_test), self.evaluate(X_020, Y_020),
            self.evaluate(X_2080, Y_2080), self.evaluate(X_80100, Y_80100)]


def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors


def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

def read_node_degree(filename):
    fin = open(filename, 'r')
    D = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        dd = l.strip().split(' ')
        D[dd[0]] = int(dd[1])
    fin.close()
    return D

vector_file = 'myresult/' + data_name + '_' + model_name + '_emd.txt'
label_file = 'mydata/' + data_name + '_label.txt'
degreeRank_file = 'mydata/' + data_name + '_degreeRank.txt'

vectors = load_embeddings(vector_file)

nams = ["micro", "macro"]
myresults = [{}, {}, {}, {}]
for tt in range(4):
    for j in nams:
        myresults[tt][j] = 0

X, Y = read_node_label(label_file)
D = read_node_degree(degreeRank_file)

print("Training classifier using {:.2f}% nodes...".format(clf_ratio*100))
for i in range(turnnum):
    clf = Classifier(vectors=vectors, clf=LogisticRegression())
    myresult = clf.split_train_evaluate(X, Y, clf_ratio, D)
    for tt in range(4):
        for j in nams:
            myresults[tt][j] += myresult[tt][j]

titles = ['All', '<=20%', '20%~80%', '>80%']
for tt in range(4):
    print('{}\t{}'.format(myresults[tt]['micro']/turnnum, myresults[tt]['macro']/turnnum))

