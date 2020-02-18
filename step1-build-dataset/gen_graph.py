from collections import defaultdict as dd
import numpy as np
import pickle as pkl
import gc
import sys
import os
import snap

seed = 1
np.random.seed(seed)
p = 0.008394918464320346 # 稀疏度

if len(sys.argv) < 3:
    print("Usage python gen_sparse_x.py dataset_name factor")
    exit(0)

dataset_name = sys.argv[1]
dataset_dir = dataset_name + "/raw"
factor = int(sys.argv[2])

if os.path.isdir(dataset_dir):
    pass
else:
    os.makedirs(dataset_dir)

nodes, edges = 52910.0529, 100000
nodes = int(nodes * factor)
edges = edges * factor

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


