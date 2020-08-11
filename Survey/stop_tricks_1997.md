# [Early Stopping - but When?]()

## 为什么使用early stopping?

解决深度学习的过拟合问题有两种方式：一是减小参数空间的维度，二是减小参数维度的有效大小.

early stopping属于方式二的策略之一，其他还有weight decay等；方式一的代表方法有greedy constructive learning, weight sharing等

## early-stoping技术

### 基本策略

1. 将训练数据划分为训练集和验证集，比如2:1
2. 在训练集上训练，每隔一段时间（比如5轮）在验证集上评估
3. 当验证集的error高于上一次的error时停止训练
4. 使用最后一次的权重作为训练阶段的结果

问题：实际情况中，验证集error曲线往往不止一个局部最小值

### 优化
