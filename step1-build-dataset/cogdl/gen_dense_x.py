from collections import defaultdict as dd
import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix
import gc
import sys
import os

seed = 1
np.random.seed(seed)

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
#nodes, edges, factor = 2708, 5429, 1
feature_dim, classes = 32, 10
nodes = int(nodes * factor)
edges = edges * factor
n_x = 20 * classes * factor # x
n_t = 1000 * factor # tx

# nodes, n_x, n_t, feature_dim = 19717, 180, 1000, 32
n_r = nodes - n_x - n_t # allx - x

data = []
row = []
col = []
for i in range(0, n_x):
    for j, v in enumerate(np.random.randint(0, 2, feature_dim)):
        row.append(i)
        col.append(j)
        data.append(v)

x = csr_matrix((data, (row, col)), shape=(n_x, feature_dim))
with open(dataset_dir + "/ind.{}.x".format(dataset_name), "wb") as f:
    pkl.dump(x, f)

print("save x successful!")
del x
gc.collect()

for i in range(n_x, n_x + n_r):
    for j, v in enumerate(np.random.randint(0, 2, feature_dim)):
        row.append(i)
        col.append(j)
        data.append(v)

allx = csr_matrix((data, (row, col)), shape=(n_x + n_r, feature_dim))
with open(dataset_dir + "/ind.{}.allx".format(dataset_name), "wb") as f:
    pkl.dump(allx, f)

print("save allx successful!")

del allx, data, row, col
gc.collect()

data = []
row = []
col = []
for i in range(n_t):
    for j in np.random.randint(0, 2, feature_dim):
        row.append(i)
        col.append(j)
        data.append(1)
tx = csr_matrix((data, (row, col)), shape=(n_t, feature_dim))
with open(dataset_dir + "/ind.{}.tx".format(dataset_name), "wb") as f:
    pkl.dump(tx, f)
print("save tx successful!")
del tx
gc.collect()

