高效图神经网络训练系统实现

## 图神经网络训练系统瓶颈分析

已完成，源代码见pyg_gnns, pyg_analysis

论文：Empirical Exploration of the Performance Bottleneck in Graph Neural Network Training, Future Generation Computer System在投

关键发现：
1. 边计算是核心性能瓶颈
2. 采样技术是解决大规模图的关键手段，但是还不够高效

## Training阶段的优化

动机：前期论文实验结果表明采样技术是解决大规模图训练的唯一手段

### Sample技术优化

1. 调研已有的采样算法工作 <font color=red>(8月8日)</font>
    - 阅读采样算法论文+论文笔记: 重点关注是否有理论说明采样算法可以加快收敛
        - 重新搜索有哪些采样论文,即论文阅读列表
            - GraphSAINT
            - ClusterGCN
            - AGSN
            - Adam
            - FastGCN
            - GraphSAGE
2. 确定使用哪些采样算法进行实现
3. 重新验证采样算法能够对时间产生收敛
    - 补充epoch轮数下，各个算法的收敛情况（见文件）
4. 优化以后的Sample算法实现（11月份之前搞定)
    - 确定细节部分
    - 验收标准：提交给PyG代码
5. 探索（1月份前搞定）
    - 并行化多个Sample是否可以加速收敛？
        - 有效性验证：收敛越快，速度越快；论文验证，实验验证
    - BatchSize越小，异步越快，速度越快；BatchSize越大，精度越高
        - 这两个点是否可以进行平衡

### 基本算子优化

为某类算子实现共性的算子，提供某类共性算子

## Inference阶段的优化

Inference阶段时，模型都是已经训练好的模型；重点关注的就是模型的执行
