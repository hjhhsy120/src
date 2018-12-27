# src
ne&amp;sample
### Changes to OpenNE
增加了MH-Walk

修改__main__.py使得LINE不会在每个epoch之后都做test，只在最后做一次

### nesampler
修改了__main__.py, line.py, node2vec.py，增加了trainer.py

trainer是根据line的二阶部分修改而成的，根据权重建立alias表，生成负采样表，做mini-batch梯度下降

line只保留了二阶部分，直接调用trainer

node2vec包含了deepwalk和node2vec算法，先调用walker（未改动）采样路径，再用myparser函数生成samples，最后调用trainer。

问题1：myparser有两种方式：

1. 每个点对都按权重1.0加入samples

2. 统计点对出现次数作为权重

不确定哪一种方式更好。当前采用第一种

问题2：细节问题，点对是否可以包含(v, v)这样自己到自己的点对？当前没有包含。

问题3：使用trainer，在email数据集上表现和原来差不多（不论parser是哪种方式），但是在cora和blogcatalog上表现很差。

尝试1：修改optimizer。原先是adam，考虑改成SGD，结果在email上都不收敛了（loss基本不变）

尝试2：调整学习率。没找到更好的学习率。

尝试3：取消batch的随机化，调整negative_ratio。似乎影响不大

尝试4：修改batch_size。没有提升cora上的效果。

修改前的超参数可参考line0.py（或者openne里的line.py）

### Other files
myresult是存放embedding结果和其他输出信息（info）的

mydata里面是数据集，包括edge, label, degreeRank(度数排序), info(基本信息), partinfo(连通性)，以及相关代码

mytest2是存放5次取平均的测试结果的

mycla是5次取平均的代码，含度数分段

combiner是把node2vec的grid search结果整合到一起的代码

