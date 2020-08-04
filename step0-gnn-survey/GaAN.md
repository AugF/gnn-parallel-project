## GaAN论文阅读笔记

摘要：

我们propose一个新的network architecture, Gated Attention Networks.

不想传统地multi-head attention mechanism机制，它equally consumes所有attention head's importance. 我们秒速了GaAN再inductive node classification问题在大规模图上。更多地，GaGN我们建立了GGRU单元来address traffic speed forecasting problem. 额外的实验在三个真实数据集上取得了效果。

介绍：

很多关键机器学习任务。这些问题的困难在于如何需要正确的方式来表达和挖掘图潜在的结构信息。传统地，这可以通过进行graph很多数据集的统计比如degree和centrality（中心），使用graph kernel, 或者提出人类特征工程特征。

近期的研究比如

整个pipeline思路：

utils.py: 主要是预处理文件
layers.py: 每一层的文件
models.py: 模型文件，其中特有的参数是按照论文中来的
train.py: 训练文件

layers:
__init__ 初始化，接受输入参数

reset_parameters, 
```
class gcnlayer(Module):
    def __init__(self, in_feature, out_feature, bias=False):
```     