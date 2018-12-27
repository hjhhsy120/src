# src
ne&amp;sample
### Changes to OpenNE
增加了MH-Walk
修改__main__.py使得LINE不会在每个epoch之后做test
### nesampler
修改了__main__.py, line.py, node2vec.py，增加了trainer.py
trainer是根据line的二阶部分修改而成的
### Other files
myresult是存放embedding结果和其他输出信息（info）的
mydata里面是数据集，包括edge, label, degreeRank(度数排序), info(基本信息), partinfo(连通性)，以及相关代码
mytest2是存放5次取平均的测试结果的
mycla是5次取平均的代码，含度数分段
combiner是把node2vec的grid search结果整合到一起的代码
