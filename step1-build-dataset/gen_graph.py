from collections import defaultdict as dd
import numpy as np
import pickle as pkl
import gc
import sys

seed = 1
p = 0.008394918464320346 # 稀疏度
dataset_name = "large"

nodes, edges, factor = 52910.0529, 100000, 1000
#nodes, edges, factor = 2708, 5429, 1
features, classes = 1500, 10
nodes = int(nodes * factor)
edges = edges * factor
n_x = 20 * classes * factor # x
n_t = 1000 * factor # tx
n_r = nodes - n_x - n_t # allx - x

feature_dim = int(features * p) # 多少维的特征是有效的

np.random.seed(seed)
Rnd = snap.TRnd()
Graph = snap.GenRMat(nodes, edges, 0.6, 0.1, 0.15, Rnd)

graph = dd(list)
for e in Graph.Edges():
    a, b = e.GetSrcNId(), e.GetDstNId()
    graph[a].append(b)
    graph[b].append(a)

with open(dataset_name + "/ind.{}.graph".format(dataset_name), "wb") as f:
    pkl.dump(graph, f)

del graph
gc.collect()

print("save graph successful!")


