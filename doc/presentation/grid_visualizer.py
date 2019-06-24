import numpy as np
import csv

from libconform import CP
from libconform.ncs import NCSDecisionTree, NCSKNearestNeighbors

X, y = [], []

with open("blue_points.csv") as b:
    data = list(csv.reader(b, delimiter=","))
    X += [[float(x) for x in row] for row in data]
    y += [0 for row in data]

with open("red_points.csv") as r:
    data = list(csv.reader(r, delimiter=","))
    X += [[float(x) for x in row] for row in data]
    y += [1 for row in data]

X, y = np.array(X), np.array(y)

idx = np.arange(len(X))
np.random.shuffle(idx)

X, y = X[idx], y[idx]

ncs = NCSKNearestNeighbors(n_neighbors=1)
#ncs = NCSDecisionTree()
cp = CP(ncs,[])
cp.train(X, y)

step = 0.02
x0 = 0.0
x1 = 0.0

X_test = []
while round(x0,2) <= 1.0:
    while round(x1,2) <= 1.0:
        X_test.append([x0, x1])
        x1 += step
    x1 = 0.0
    x0 += step

X_test = np.array(X_test)

labels, sig_lvls = cp.predict_best(X_test)

eps = [0.02,0.03,0.04,0.05]

sets = {e: {l: [] for l in [0,1]} for e in eps}

for l, s, x in zip(labels, sig_lvls, X_test):
    for e in eps:
        if s < e:
            sets[e][l].append(list(x))
            break

for e in sets:
    for i in [0,1]:
        with open("data/{}_{}.csv".format(e,i),"w") as f:
            w = csv.writer(f, delimiter=",")
            w.writerows(sets[e][i])
