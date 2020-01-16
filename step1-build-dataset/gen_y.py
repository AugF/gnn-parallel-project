import numpy as np
import pickle as pkl
import gc
import sys
import os

seed = 1
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
#nodes, edges, factor = 2708, 5429, 1
features, classes = 1500, 10
nodes = int(nodes * factor)
n_x = 20 * classes * factor # x
n_t = 1000 * factor # tx
n_r = nodes - n_x - n_t # allx - x

np.random.seed(seed)

# ind.dataset.ally + ty
y = np.zeros((n_x, classes))
for i in range(n_x):
    y[i][i % 10] = 1
ry = np.zeros((n_r, classes))
labels_r = np.random.randint(0, 10, (n_r, 1))
for i in range(n_r):
    ry[i][labels_r[i]] = 1
ally = np.vstack((y, ry))

with open(dataset_dir + "/ind.{}.y".format(dataset_name), "wb") as f:
    pkl.dump(y, f)
with open(dataset_dir + "/ind.{}.ally".format(dataset_name), "wb") as f:
    pkl.dump(ally, f)
del ally
del y
print("save ally, y successful!")
gc.collect()

ty = np.zeros((n_t, classes))
labels_t = np.random.randint(0, 10, (n_t, 1))
for i in range(n_t):
    ty[i][labels_t[i]] = 1

with open(dataset_dir + "/ind.{}.ty".format(dataset_name), "wb") as f:
    pkl.dump(ty, f)
del ty
print("save ty successful!")
gc.collect()

with open(dataset_dir + "/ind.{}.test.index".format(dataset_name), "w") as f:
    for i in range(nodes - n_t, nodes):
        f.write(str(i) + "\n")
