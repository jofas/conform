from sklearn.datasets import load_iris
from time import time

import numpy as np

from cp import CP, CPMetrics
from nc_scores import NC1NN

from tests import IRIS_SUBSET_X, IRIS_SUBSET_Y

# TODO:
#       Predicted class
#       make NC1NN beautiful and readable
#       benchmark.py
#       tests.py with integration tests
#       usps test
#       py <-> julia / py <-> rust (rayon - parallel iter)

def iris_prediction_tutorial_cp():
    print("\nTutorial")

    X  = IRIS_SUBSET_X[:24]
    y  = IRIS_SUBSET_Y[:24]
    x_ = IRIS_SUBSET_X[24]
    y_ = IRIS_SUBSET_Y[24]

    epsilons = [0.05, 0.08, 0.33]

    start = time()
    cp = CP(NC1NN(), epsilons, np.array([0,1]))
    cp.update(X, y)
    res = cp.predict(x_)
    print(res)
    print("time: " + str(time() - start))

def basic_iris():
    print("\nBasic Iris")

    iris = load_iris()

    # only setosa and versicolor
    iris_reduced_X = iris['data'][:100]
    iris_reduced_y = iris['target'][:100]

    indices = np.arange(iris_reduced_X.shape[0])
    np.random.shuffle(indices)

    X = iris_reduced_X[indices][:25]
    y = iris_reduced_y[indices][:25]

    n, p = X.shape
    epsilons = [0.08]

    start = time()
    cp = CP(NC1NN(), epsilons, np.array([0,1]))
    res = cp.score_online(X, y)
    print(res)
    print("time:", time() - start)

    # long version score_online
    '''
    cp = CP(NC1NN(), epsilons, np.array([0,1]))
    res = CPMetrics(epsilons)
    for i in range(n):
        res += cp.score(X[i], y[i])
        cp.update(X[i], y[i])

    print(res)
    '''

def iris_1000_tutorial_cp():
    print("\n1000 Prediction Regions")

    iris = load_iris()

    # only setosa and versicolor
    iris_reduced_X = iris['data'][:100]
    iris_reduced_y = iris['target'][:100]

    # only sepal length
    iris_reduced_X = np.delete(iris_reduced_X, [1,2,3], 1)

    indices = np.arange(iris_reduced_X.shape[0])

    epsilons = [0.08]

    res = CPMetrics(epsilons)
    t   = 0

    for i in range(1000):
        np.random.shuffle(indices)

        X  = iris_reduced_X[indices][:24]
        y  = iris_reduced_y[indices][:24]
        x_ = iris_reduced_X[indices][24]
        y_ = iris_reduced_y[indices][24]

        start = time()

        cp = CP(NC1NN(), epsilons, np.array([0,1]))
        cp.update(X, y)

        res += cp.score(x_, y_)
        t   += time() - start

    print(res)
    print("time:", t)

def main():
    iris_prediction_tutorial_cp()
    basic_iris()
    iris_1000_tutorial_cp()

if __name__ == '__main__':
    main()
