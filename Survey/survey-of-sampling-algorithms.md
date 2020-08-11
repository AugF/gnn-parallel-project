---
title: 现有GNN采样算法调研
author: Yunpan Wang
---

## 一、GraphSAINT
### 1.论文资料
作者：曾涵清博士，南加州大学
论文：在 ICLR 2020 上发表了GraphSAINT: Graph Sampling Based Inductive Learning Method
代码：https://github.com/GraphSAINT/GraphSAINT

### 2.传统GNN挑战：邻居爆炸（Neighbor Explosion）
邻居爆炸：
GNN会不断地聚合图中相邻节点的信息，则L-层的GNN中每个目标节点都需要聚合原图中L-hop以内的所有节点信息。在大图中，邻节点的个数可以随着L指数级增长。

邻点爆炸式增长，使得GNN的minibatch训练极具挑战性。
受显存的因素限制，GNN在不损失精度的前提下，难以高效地被拓展到大于两层的模型（否则指数爆炸）。

### 3.现有方法：图采样

Layer-wise Sampling：
邻居爆炸：在矩阵采样多层时，假设每层采样n个邻居，则会导致n^2级别的节点扩充速度。
领接矩阵稀疏：在矩阵采样的过程中，会导致邻接矩阵稀疏化，丢失一些本来存在的边。
时间耗费高：每一层卷积都要采样，这就导致计算的时间耗费。
前人工作提出了众多基于采样邻居节点的方法，并利用多类型的聚合函数提高模型的表达能力。大部分现有工作都在探索如何通过对GNN每层的节点进行采样，以降低训练成本。

Graph-wise Sampling：
从原图上采样一个子图下来，在这个子图上做局部的、完全展开的GCN。
解决了邻居采样的指数化问题，而且可以对采下来的子图进行直接的并行化，就大大的改进了效率。即，可以在preprocess阶段提前采样图，并且可以进行mini batch的加速。
但是这样的采样往往会丢失一些信息。
![](survey-of-sampling-algorithms\GraphSAINT\Fig1.png)

### 4.GraphSAINT：截然不同的采样的视角
属于Graph sampling。
从原图中采样子图，在子图上使用GCN学习。
基于极其轻量级的子图采样算法，同时实现了在准确率和复杂度上的显著提升。
提出了适用于大图以及深层网络的，通用的训练框架。
在标准的Reddit数据集上，以100倍的训练时间提速，提高了1%以上的准确率。在这里插入图片描述
#### 4.1.算法流程
将全图进行多次采样，在得到的sub-graph上进行全GCN，然后将多个sub-graph的信息融合起来。
![](survey-of-sampling-algorithms\GraphSAINT\Alg1.png)

#### 4.2.子图采样
![](survey-of-sampling-algorithms\GraphSAINT\sampler.png)
本文的采样是基于节点的连通性：
1. 相互影响较大的节点应在同一子图中采样，intra sub-graph的节点是具有强联系的，但这就引入了采样的bias。理想的SAMPLE要求衡量节点连接的联合信息以及属性。但这种算法可能具有很高的复杂度，所以，为简单起见，从图连接性角度定义“影响力”，并设计基于拓扑的采样器。
2. 为了消除保留图连通性特征的采样器引入的偏差，作者引入自行设计的归一化技术，以消除偏差，使得估计量无偏。归一化系数通过pre-processing估计出来。
3. 每条边的采样概率均不可忽略，使得神经网络能够探索整个特征和标签空间。
4. 考虑variance reduction（降低采样方差），旨在得到最小方差的边采样概率。

重点是估计每个节点、边、子图的采样概率。

- 节点采样概率：
![](survey-of-sampling-algorithms\GraphSAINT\node_pro.png)

- 边采样概率:
![](survey-of-sampling-algorithms\GraphSAINT\edge_pro.png)

- 子图采样概率：
![](survey-of-sampling-algorithms\GraphSAINT\subgraph_pro.png)

> 这篇文章的采样sub-graph的概念应该是介于AS-GCN的“有条件的采样全图”和ClusterGraph的“将图聚类后采样”之间。由于采样的存在，FastGCN, AS-GCN等在不同的layer计算的是不同的图结构，而GraphSAINT, ClusterGraph可以看做在不同layer都对同一个graph进行特征提取。

#### 4.3.实验结果：优于GCN, SAGE…
![](survey-of-sampling-algorithms\GraphSAINT\Table2.png)

原因分析：
GCN相当于full gradient descent，没有batch泛化好。
GraphSAGE采样batch方差大，收敛性不好。

### 参考文献
- https://zhuanlan.zhihu.com/p/107107009
- https://rman.top/2020/05/20/GraphSAINT%E4%B8%80%E7%A7%8D%E6%97%A0%E5%81%8F%E7%9A%84%E5%9B%BE%E9%87%87%E6%A0%B7%E6%96%B9%E6%B3%95/
- https://www.readercache.com/q/EP8b9Dpq4ZAw0ENEKGRr1zGKX5MRjeNY

## 二、Cluster-GCN

### 1. 摘要
图卷积网络（GCN）已经成功地应用于许多基于图形的应用，然而，大规模的GCN的训练仍然具有挑战性。目前基于SGD的算法要么面临着随GCN层数呈指数增长的高计算成本，要么面临着保存整个图形和每个节点的embedding到内存的巨大空间需求。本文提出了一种新的基于图聚类结构且适合于基于SGD训练的GCN算法 — Cluster-GCN。

Cluster-GCN的工作原理如下：在每个步骤中，它对一个与通过用图聚类算法来区分的密集子图相关联的一组节点进行采样，并限制该子图中的邻居搜索。这种简单且有效的策略可以显著提高内存和计算效率，同时能够达到与以前算法相当的测试精度。

为了测试算法的可扩展性，作者创建了一个新的Amazon2M数据集，它有200万个节点和6100万个边，比以前最大的公开可用数据集（Reddit）大5倍多。在该数据上训练三层GCN，Cluster-GCN比以前最先进的VR-GCN（1523秒vs 1961秒）更快，并且使用的内存更少（2.2GB vs 11.2GB）。此外，在该数据上训练4层GCN，Cluster-GCN可以在36分钟内完成，而所有现有的GCN训练算法由于内存不足而无法训练。此外，Cluster-GCN允许在短时间和内存开销的情况下训练更深入的GCN，从而提高了使用5层Cluster-GCN的预测精度，作者在PPI数据集上实现了最先进的test F1 score 99.36，而之前的最佳结果是98.71。

### 2.背景介绍
图卷积网络（GCN）[9]在处理许多基于图的应用中日益流行，包括半监督节点分类[9]、链路预测[17]和推荐系统[15]。对于一个图，GCN采用图卷积运算逐层地获取节点的embedding：在每一层，要获取一个节点的embedding，需要通过采集相邻节点的embedding，然后进行一层或几层线性变换和非线性激活。最后一层embedding将用于一些最终任务。例如，在节点分类问题中，最后一层embedding被传递给分类器来预测节点标签，从而可以对GCN的参数进行端到端的训练。

由于GCN中的图卷积运算（operator）需要利用图中节点之间的交互来传播embeddings，这使得训练变得相当具有挑战性。不像其他神经网络，训练损失可以在每个样本上完美地分解为单独的项（decomposed into individual terms），GCN中的损失项(例如单个节点上的分类损失)依赖于大量的其他节点，尤其是当GCN变深时。由于节点依赖性，GCN的训练非常慢，需要大量的内存——反向传播需要将计算图上的所有embeddings存储在GPU内存中。

### 3.现有算法的缺陷

为了证明开发可扩展的GCN训练算法的必要性，文中首先讨论了现有方法的优缺点，包括：内存需求、每个epoch的时间、每个epoch收敛速度。

这三个因素是评估训练算法的关键。注意，内存需求直接限制了算法的可扩展性，后两个因素结合在一起将决定训练速度。在接下来的讨论中，用N为图中的节点数，F为embedding的维数，L为分析经典GCN训练算法的层数。

GCN的第一篇论文提出了全批次梯度下降（Full-batch gradient descent）。要计算整个梯度，它需要存储所有中间embeddings，导致O(NFL)内存需求，这是不可扩展的。
GraphSAGE中提出了Mini-batch SGD。它可以减少内存需求，并在每个epoch执行多次更新，从而加快了收敛速度。然而，由于邻居扩展问题，mini-batch SGD在计算L层单个节点的损失时引入了大量的计算开销。
VR-GCN提出采用variance减少技术来减小邻域采样节点的大小。但它需要将所有节点的所有中间的embeddings存储在内存中，从而导致O(NFL)内存需求。

### 4.朴素Cluster-GCN

作者定义了“Embedding utilization”的概念来表达计算效率。如果节点i在第l层的embedding在计算第l+1层的embeddings时被重用了u次，那么就说相应的的embedding utilization是u。

下表中总结了现有GCN训练算法相应的时间和空间复杂度。显然，所有基于SGD的算法的复杂度都和层数呈指数级关系。对于VR-GCN，即使r很小，也会产生超出GPU内存容量的巨大空间复杂度。

本文提出的的Cluster-GCN算法，它实现了两全其美的效果：即每个epoch和full gradient descent具有相同的时间复杂度， 同时又能与朴素GD具有相同的空间复杂度。

![](survey-of-sampling-algorithms\ClusterGCN\table1.png)

文中的Cluster-GCN技术是由以下问题驱动的：在mini-batch SGD更新中，我们可以设计一个batch和相应的计算子图来最大限度地提高embedding utilization吗？文中使用了图聚类算法来划分图。图聚类的方法，旨在在图中的顶点上构建分区，使簇内连接远大于簇间连接，从而更好地捕获聚类和社区结构。

下图展示了两种不同的节点分区策略：随机分区和clustering分区。可以看到，cluster-GCN可以避免大量的邻域搜索，并且集中在每个簇中的邻居上。作者使用随机分割和Metis聚类方法将图分成10个部分。然后使用一个分区作为一个batch来执行SGD更新。在相同的时间段内，使用聚类划分可以获得更高的精度。这表明使用图聚类是很重要的，分区不应该随机形成。

![](survey-of-sampling-algorithms\ClusterGCN\fig1.png)

**随机多聚类**
尽管朴素Cluster-GCN实现了良好的时间和空间复杂度，但仍然存在两个潜在问题：
- 图被分割后，一些连接被删除。因此，性能可能会受到影响。
- 图聚类算法往往将相似的节点聚集在一起，因此聚类的分布可能不同于原始数据集，从而导致在执行SGD更新时对 full gradient的估计有偏差。

为了解决上述问题，文中提出了一种随机多聚类方法，在簇接之间进行合并，并减少batch间的差异（variance）。作者首先用一个较大的p把图分割成p个簇V1,...,Vp，然后对于SGD的更新重新构建一个batch B，而不是只考虑一个簇。随机地选择q个簇，定义为t1,...,tq ,并把它们的节点包含到这个batch B中。此外，在选择的簇之间的连接也被添加回去。作者在Reddit数据集上进行了一个实验，证明了该方法的有效性。
![](survey-of-sampling-algorithms\ClusterGCN\fig3.png)
![](survey-of-sampling-algorithms\ClusterGCN\fig4.png)

**算法流程**
![](survey-of-sampling-algorithms\ClusterGCN\alg1.png)

### 5.实验结果
文中评估了所提出的针对四个公共数据集的多标签和多类分类两个任务的GCN训练方法，数据集统计如表3所示。Reddit数据集是迄今为止为GCN所看到的最大的公共数据集，为了测试GCN训练算法在大规模数据上的可扩展性，作者基于Amazon co-purchase network构建了一个更大的图Amazon2M，包含超过200万个节点和6100万条边。
![](survey-of-sampling-algorithms\ClusterGCN\table3.png)
作者比较了不同层次GCNs的VRGCN在训练时间、内存使用和测试准确度(F1分数)方面的差异。从表中可以看出
- 训练两层时VRGCN比Cluster-GCN快，但是当增加一层网络，却慢于实现相似准确率的Cluster-GCN；
在内存使用方面，VRGCN比Cluster-GCN使用更多的内存(对于三层的情况5倍多)。
- 当训练4层GCN的时候VRGCN将被耗尽，然而Cluster-GCN当增加层数的时候并不需要增加太多的内存，并且Cluster-GCN对于这个数据集训练4层的GCN将实现最高的准确率。

![](survey-of-sampling-algorithms\ClusterGCN\table8.png)

### 参考文献
- https://zhuanlan.zhihu.com/p/85950252

## 三、VRGCN

### 1.摘要

GCN之前的工作都在降采样邻居的数目，但是这些算法没有保证算法的收敛性，而且他们每个节点的感受野仍然在非常大的范围。本文中，作者使用了基于control varite的算法允许采样任意小的邻居大小，并且证明了该算法可以收敛到GCN的局部最优解。实验结果表明，本文提出的算法在neighbor size设置为2的情况达到了和精确的算法相似的收敛性。在Reddit数据集上，训练时间比GCN, GraphSAGE, FastGCN快了7倍以上

### 2.背景

![](survey-of-sampling-algorithms\VRGCN\fig1.png)

### 3.算法

**control variate based algorithm**
当计算$\sum_{v \in n(u)} P_{uv} h_v^{(l)}$时，需要一层一层向前递归计算，哈需要上一层网络的激活函数，这样的开销太大了。作者提出的想法是将每一层的$h_v^{(l)}$计算出来后搞一个$\overline{h}_v^{(l)}$作为近似，我们这里设定$\triangle h_v^{(l)} = h_v^{(l)} - \overline{h}_v^{(l)}$
![](survey-of-sampling-algorithms\VRGCN\control_variate.png)

**processing strategy**
$Z^{(l+1)} = P Dropout_p({H^{(l)}}) W^{(l)}$和$Z^{(l+1)} = Dropout_p(P {H^{(l)}}) W^{(l)}$

**算法流程**
![](survey-of-sampling-algorithms\VRGCN\alg1.png)
![](survey-of-sampling-algorithms\VRGCN\alg2.png)
![](survey-of-sampling-algorithms\VRGCN\alg3.png)

### 4.实验结果

1. 实验精度
![](survey-of-sampling-algorithms\VRGCN\table3.png)
- M0: Dropout in semi-GCN, without neighbor sampling
- M1: Dropout in this paper, without neighbor sampling
- M1+PP: $D^{l}$ = 20, with neigbor sampling and dropout in paper.

2. 运行时间和收敛性
![](survey-of-sampling-algorithms\VRGCN\table4.png)

## 四、Adam
在这篇论文中，研究者设计了一种自适应的逐层采样方法，可加速图卷积网络的训练。通过自上而下地构建神经网络的每一层，根据顶层的节点采样出下层的节点，可使得采样出的邻居节点被不同的父节点所共享并且便于限制每层的节点个数来避免过度扩张。更重要的是，新提出的逐层采样的方法是自适应的，并且能显式地减少采样方差，因此能强化该方法的训练。实验证明在有效性和准确性上优于其他基于采样的方法:GraphSAGE和FastGCN。

文中还进一步提出了一种新颖的跳越（skip）连接方法，可用于加强相隔比较远的节点之间的信息传播。在几个公开的数据集上进行了大量实验，实验结果表明，跳跃连接进一步提高了算法的收敛速度和最终分类精度。

### 1.简介

当前在图神经网络上的最大问题就是可扩展性问题。计算卷积要求递归地逐层扩展邻居，这就要求很大的计算量和巨大的内存。即使是单个节点，如果图是密集的或幂律的，由于逐层的邻域扩展，它也会很快覆盖图的很大一部分。传统的mini-batch训练即使在batch size很小时也无法加快卷积计算的速度，因为每个batch训练都会涉及大量的顶点。

为了避免邻居的过度扩张问题,文中采用控制每一层采样大小来加快训练。通过自顶向下的方式建立网络层，底层的节点的采样通过概率依赖于顶层的节点。这种分层采样的方式是有效的，因为

- 可以重用采样的邻居信息，因为下层的节点是可见的并且被上层不同的父节点所共享
- 很容易确定每个层的size，以避免邻域的过度扩展，因为底层的节点是作为一个整体进行采样的

![](survey-of-sampling-algorithms\Adam\fig1.png)

- 上图表示了不同算法创建的网络结构
- 图a：逐节点采样
- 图b：逐层采样
- 图c: 考虑skip-connection的模型
- 图中红色节点都有两个父节点
- 在逐节点采样中，每个父节点的领域不会被其他父节点看到，因此领域和其他父节点之间的连接是未使用的。相比之下，对于逐层采样策略，所有的领域由父层中的节点共享，因此所有层间连接都被利用

**创新点**
- 创建了一个逐层采样的方式加速GCN模型，这种方式的层之间的信息是共享的并且采样的节点的数量是可控的
- 这种逐层采样的采样方式是自适应的并且在训练阶段显式地由方差降低来确定
- 提出了一种简单而有效的方法，通过在两个层之间建立skip-connection来保持second-order proximity

### 2. 相关工作
两种基于采样的方法目标都是实现在图上的快速表示学习：GraphSAGE和FastGCN。

另外一个工作在论文“ Stochastic training of graph convolutional networks with variance reduction”中，提出了一种基于控制变量的方法（control-variate-based method），但是这种采样过程也是node-wise的，需要节点的历史激活信息。
### 3. Notations and Preliminaries

![](survey-of-sampling-algorithms\Adam\figure1.png)

### 4. 自适应采样

#### Node-Wise Sampling

![](survey-of-sampling-algorithms\Adam\figure2.png)
​
#### Layer-Wise Sampling

![](survey-of-sampling-algorithms\Adam\figure3.png)

#### Explicit Variance Reduction
![](survey-of-sampling-algorithms\Adam\figure4.png)
![](survey-of-sampling-algorithms\Adam\figure4_2.png)
![](survey-of-sampling-algorithms\Adam\figure4_3.png)

### 5. Preserving Second-Order Proximities by Skip Connections
![](survey-of-sampling-algorithms\Adam\figure5.png)

### 6. 讨论
**与其他方法的讨论**
![](survey-of-sampling-algorithms\Adam\figure6.png)
**考虑注意力机制**
![](survey-of-sampling-algorithms\Adam\figure6_2.png)

### 7. 实验结果
![](survey-of-sampling-algorithms\Adam\table1.png)

![](survey-of-sampling-algorithms\Adam\fig2.png)

![](survey-of-sampling-algorithms\Adam\fig3.png)

## 五、LGCN
### 摘要
在解决full-batch训练带来的高计算和内存开销问题上，node-wise的邻居采样方法递归地采样固定数目的邻居，导致它的计算开销指数级别地上升;layer-wise的重要性采样方法抛弃了neighbor-dependent的限制，导致采样出的节点有sparse connection的问题；为了解决这两个问题，本文提出一个有效的采样方法，LAyer-Dependent ImportancE Sampling(LADIES). 基于在上层采样到的顶点，LDIES选择他们的邻居节点，重建了二分

## 六、FastGCN

![](survey-of-sampling-algorithms\FastGCN\figure1.png)
![](survey-of-sampling-algorithms\FastGCN\figure2.png)
![](survey-of-sampling-algorithms\FastGCN\figure3.png)
### 参考文献
https://www.jianshu.com/p/48b526d29f0a

## 七、GraphSAGE