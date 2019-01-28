# src

### 热力图
热力图：第i行第j列的色块对应vertex i和context j的出现次数，次数过大的做了最大值限制。i和j的数值是相应节点的度数从小到大的排名（相当于将节点按度数从小到大排了序），超过5000个点则把所有点排名乘5000除以点数取整，以保证复杂度不会过大。现有email, cora, blogcatalog在deepwalk, lpwalk, app上的结果，node2vec做了email上grid search 0.25, 1.0, 4.0，在mypic文件夹下。

相关代码：nesample/trainer在得到所有sample之后按每行“v c weight”格式输出到aaa.txt然后退出；pic.py读取sample信息和度数排序、绘制热力图；zrun_nesampler.bat是它的批处理文件。

### vc-sampling

#### 详见vcsample文件夹下的readme

相关代码在vcsample文件夹下，现在主要是vctrainer.py和修改过的app.py，以及有点问题的deepwalk.py

vctrainer是训练的框架，调用model_v的sample_v(batch_size)和model_c的sample_c(h)产生vc对，用tensorflow训练。negative sampling的实现也在这个代码里面。

sample_v是采样“中心点”的generator，生成一个epoch的各个batch，通过yield语句每次输出一个batch的v。sample_c函数的参数是sample_v输出的所有“中心点”，它根据v采样context，得到context序列并输出。输出的都是lookup之后的标号，不是原图的标号。

对app而言，sample_v就是随机打乱节点顺序，然后每个节点依次取sample个，取满batch_size就yield，直到取完；sample_c就是遍历输入的序列，对每个中心点，从它出发以一定概率停止地走不超过10步，输出停止的节点lookup的标号。在email和cora上的结果不如原来的deepwalk，比原来的LINE好

关于deepwalk，我现在想先用pagerank计算平稳分布，再建立一个固定大小的列表（点数的fac倍大小），使得节点在列表的出现次数和pagerank值成正比，在sample_v中对列表做一遍random shuffle，然后依次取；采样context的时候，我想从中心点随机游走window步，将路径上的点都加入context。（实际操作的时候，记录了每个中心点对应的已走步数和当前位置，这样可以不受batch“隔断”的影响）。不过在email上的运行结果比LINE差，在Cora上的结果比APP稍差、比LINE好，不知道是不是哪里有问题。。

### Changes to OpenNE
增加了MH-Walk

修改__main__.py使得LINE不会在每个epoch之后都做test，只在最后做一次

增加了lpWalk

增加了APP

其中，默认iters=200, sample=200，按原代码实际的采样个数是node_num x iters x sample，batch_size = 1, epoch = 1, 时间开销非常大。

一方面，仅仅是采样的复杂度就已经远大于其他算法了，比如deepwalk是node_num x num_paths x walk_length，而它的num_paths和walk_length都比较小(10, 80)，所以我想在APP里面也把参数调小一点。

另一方面，原代码每一步都做更新，但是在点数很多的时候，会导致非常大的更新次数。所以，除非不用tensorflow做训练，否则大概很难有效地复现APP算法。不知调大batch_size对结果会有怎样的影响。我试过batch_size=1000, epoch=10和batch_size=100, epoch=1，结果差不多

除上述差异外，原代码的学习率衰减在这里没有实现。

当前参数：iters=10, sample=50, batch_size=1000, epoch=10, learning_rate=0.001(adam) 其他参数按原代码：停止概率jump_factor=0.15, 最大步数step=10, negative_ratio=5

修复bug后，表现比deepwalk稍差，比line稍好，详见表格。

### nesampler
修改了__main__.py, line.py, node2vec.py，增加了trainer.py

trainer是根据line的二阶部分修改而成的，根据权重建立alias表，生成负采样表，做mini-batch梯度下降

line只保留了二阶部分，直接调用trainer

node2vec包含了deepwalk和node2vec算法，先调用walker（未改动）采样路径，再用myparser函数生成samples，最后调用trainer。

问题1：myparser有两种方式：

1. 每个点对都按权重1.0加入samples

2. 统计点对出现次数作为权重

不确定哪一种方式更好。当前采用第二种，否则点对数量可能过大。

问题2：细节问题，点对是否可以包含(v, v)这样自己到自己的点对？当前没有包含。

修改：现在就是按照skipgram的算法，对于窗格内除了中心点之外的点，都加进点对。

问题3：使用trainer，在email数据集上表现和原来差不多（不论parser是哪种方式），但是在cora和blogcatalog上表现很差。

尝试1：修改optimizer。原先是adam，考虑改成SGD，结果在email上都不收敛了（loss基本不变，分类效果非常差）

尝试2：调整学习率。没找到更好的学习率。

尝试3：取消batch的随机化，调整negative_ratio。似乎影响不大

尝试4：修改batch_size。没有提升cora上的效果。

尝试5：修改epoch数。即使在loss基本稳定的情况下，也基本没有提升cora上的效果。

尝试6：怀疑log_sigmoid出现NAN，做clip_by_value使得log的参数在1e-8~1.0的范围（原本是0.0~1.0）。结果没有明显变化。

尝试7：修改loss为nce_loss……结果很差。

现在使用adam, lr=0.001, batch_size=1000, batch有random, 5个epoch（main的默认），有clip_by_value，loss仍然是log_sigmoid

修复bug：原先少了一个look_up。节点在graph的编号从0开始，但节点名称可能从1开始，所以需要look_up。

email数据集的节点名称从0开始，所以表现正常；cora从1开始，所以出现了问题。

现已修复。表现见表格。

### Other files
myresult是存放embedding结果和其他输出信息（info）的

mydata里面是数据集，包括edge, label, degreeRank(度数排序), info(基本信息), partinfo(连通性)，以及相关代码

mytest2是存放5次取平均的测试结果的

mycla是5次取平均的代码，含度数分段

combiner是把node2vec的grid search结果整合到一起的代码

