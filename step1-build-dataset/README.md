https://docs.qq.com/doc/DQ0tOaU52TFdkcGhCora dataset
## 1. 格式说明

dataset/raw/ 
对于图数据集，通常会处理为以下几个格式：
- ind.dataset.x:  train, save as scipy.csr.csr_matrix object
- ind.dataset.tx: test, save as scipy.csr.csr_matrix object
- ind.dataset.allx: train+val+else, save as scipy.csr.csr_matrix object
- ind.dataset.y: one-hot labels, save as numpy.ndarray object
- ind.dataset.ty: one-hot lables, save as numpy.ndarray object
- ind.dataset.ally: one-hot lables, save as numpy.ndarray object
- ind.dataset.graph: a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict.
- ind.dataset_str.test.index: the indices of test instances in graph, for the inductive setting as list object.

## 2. 文件说明
### gen
1. gen_graph.py: 生成graph的文件
2. gen_y.py: 生成y的文件
3. gen_sparse_x.py: 生成稀疏特征
4. gen_dense_x.py: 生成稠密特征

### examples

基本顶点，边的数量
nodes, edges = 52910.0529 * factor, 100000 * factor

medium1, factor=1
medium10, factor = 10
medium100, factor = 100
medium1000, factor = 1000
larger. factor = 10000
  
