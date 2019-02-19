# vcsample使用说明

## 命令行参数说明

### 输入输出相关

--input 输入图文件名，可以是邻接表或者边表，不可缺省

--output 输出embedding文件名；可选

--label-file 输入节点标签文件名；可选

--graph-format 输入图格式，可以是adjlist（邻接表，默认）或edgelist（边表）

--weighted 是否有权重，写了表示有权重，下同；带权图相关代码尚不完善

--directed 是否有向

--embedding-file 预训练的embedding结果，若设置该项，则除link prediction外都可跳过embedding训练，直接评测

### Embedding训练相关

--representation-size embedding的维度；默认128

--epochs 训练epoch数，默认20

--epoch-fac 一个epoch包含的节点数除以节点总数的值，默认50

--batch-size 默认1000

--lr 学习率，默认0.001

--negative-ratio 负采样数和正样例数的比值n。因为是按“一个正样例batch+n个负样例batch”轮流训练的，所以必须是非负整数；默认5

### 算法相关

--model-v 指定vertex采样方式，不可缺省

--model-c 指定context采样方式，若缺省则默认与model_v相同，相同的model只会做1次初始化

#### app

--app-jump-factor 停止的概率；默认0.15

--app-step 随机游走的最大步数；默认80

#### deepwalk

--window-size 采样context时，对应的窗格大小，默认10

### Evaluation相关

--exp-times 实验次数，适用于节点分类（多次划分训练、测试集）、节点聚类（多次计算k-means）取平均结果；默认1

#### 节点分类

--classification 是否为节点分类任务（仅当有标签时）

--clf-ratio 列表形式，每项是训练数据占节点总数的比例，用逗号分隔（不带空格）；默认0.5

#### Link Prediction

--link-prediction 是否为链接预测任务

--prop-pos 删除的边（测试的正样例）占总边数的比例；默认0.5

--prop-neg 测试用的负样例占采样的负边总数的比例；默认0.5（一般和prop_pos一样）

--prop-neg-tot 采样的负边总数和原图边数的比例，这些边将根据prop_neg划分为训练、测试两部分；默认1.0

--cached-fn 保存的图的名称，不写则不保存也不使用保存的图

#### Graph Reconstruction

--reconstruction 是否为graph reconstruction任务

--k-nbrs knn的k，默认30

#### Vertex Clustering

--clustering 是否为带聚类标签（不支持多标签）的vertex clustering任务，此任务下将固定k-means的k值，以k-means结果和标签的NMI为评测指标

--modularity 是否为modularity评测，此任务将枚举k值，是的聚类结果的modularity值最大，输出相应的k和modularity

--min-k modularity任务枚举的最小k，默认2

--max-k modularity任务枚举的最大k，默认30

## 文件说明

main 程序入口，包括基本流程代码

getmodel 修改命令行参数、增加新模型入口

graph 图的存储、转换，包括link prediction的删边、选负样例等

vctrainer 采样vc、分batch训练

classify 节点分类

link 链接预测

reconstr graph reconstruction

clustering 聚类，包括modularity

app 类似app的采样，sample_v是随机均匀选择非孤立点，sample_c是APP的随机游走

deepwalk 类似deepwalk的采样，sample_v是以pagerank为权重选点，sample_c是假定pagerank为稳定分布，用随机游走试图还原deepwalk的skipgram对context的采样

## 如何添加模型

新模型需要建立一个class，至少包含sample_v或sample_c之一；然后再getmodel中添加命令行参数和调用model的代码

sample_v是一个generator，生成一个epoch所用到的vertex，当数量达到batch_size时yield，最后如有剩余就再yield一次

sample_c是一个函数，输入一组vertex，输出根据这些vertex和model初始化信息所生成的一组context

## 评测说明

若设置了link-prediction任务，则不能做其他任务，因为会报错：

ValueError: Variable model/embeddings already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?

原因是link prediction的embedding结果不能用在之后的任务中，而做第二次embedding就会在tensorflow中做一些重复设置，导致出错。

其他任务可以同时设置多个，也可以保存embedding结果之后再逐个评测（调参）；顺序是node classification, clustering, modularity, graph reconstruction

### 节点分类

读图，做embedding，然后将节点划分为训练集和测试集，使用sklearn根据训练集的embedding向量和label训练一个分类器，考察测试集上的表现

### Link Prediction

读图，选一些不在图上的边，划分成训练集负例、测试集负例两部分；将图上的一部分边去掉，去掉的边作为测试集的正例，剩下的边作为训练集正例。用剩下的图做embedding，然后用训练集的边的embedding和label（正例1，负例0）训练一个分类器，考察测试集上的AUC。其中，边的embedding根据两个节点的embedding计算而来，计算方法包括hadamard（逐项乘），average（逐项平均），l1（逐项取差的绝对值），l2（逐项取差的平方），concat（拼接）

### Graph Reconstruction

读图，做embedding；用余弦距离表示两个节点之间有连边的概率，原先的算法是对每个节点计算它和其他节点的余弦距离并排序（复杂度 $n^2\log n$），选最大的k个（k是节点的邻居数），看有多少个是它的邻居；现在改为用KNN，准确率稍微下降了一些，复杂度降低了很多；取所有预测中正确的邻居数除以总数作为结果。不过，reconstruction结果非常差，原因未知

### Clustering

读图，做embedding；

以NMI为指标：设置k-means的k为标签数，用embedding向量做聚类，以聚类结果和实际标签的NMI值作为结果。此处不支持多标签数据

以modularity为指标：从min_k到max_k枚举k，做k-means，然后计算modularity值，选最大的一个作为结果

modularity计算：对每个类簇，计算内部节点度数之和degree_sum（有向图考虑入度加出度之和，无向图因为是按双向边存的所以代码中要除以2），记录内部边的数量edge_inside（无向图按双向边来算，所以是实际边数的2倍），计算 $\text{edge_inside}/2m - (\text{degree_sum}/2m)^2$ （其中m是边数，无向图的边只算1条），对类簇求和，即modularity值。