from libconform import CP
from libconform.ncs import NCSKNearestNeighbors

from helpers import X_train, y_train, X_test, y_test

def basic_cp(labels=[0,1,2]):
    sig_lvls = [0.01, 0.025, 0.05, 0.1]
    ncs = NCSKNearestNeighbors()
    return CP(ncs, sig_lvls, labels=labels)

def test_predict_best_in_predict():
    cp = basic_cp()
    cp.train(X_train, y_train)

    pred      = cp.predict(X_test)
    pred_best = cp.predict_best(X_test, sig_lvls = False)

    print(pred_best)

    for p, p_ in zip(pred, pred_best):
        for l in p:
            if len(l) > 0:
                assert p_ in l

def test_label_sequence_irrelevant():
    cp = basic_cp([1,2,0])
    cp.train(X_train, y_train)
    pred0 = cp.predict(X_test)

    cp = basic_cp([2,0,1])
    cp.train(X_train, y_train)
    pred1 = cp.predict(X_test)

    cp = basic_cp([2,1,0])
    cp.train(X_train, y_train)
    pred2 = cp.predict(X_test)

    for p0, p1, p2 in zip(pred0, pred1, pred2):
        for c0, c1, c2 in zip(p0, p1, p2):
            assert sorted(c0) == sorted(c1) == sorted(c2)

def test_predict_encoded():
    cp = basic_cp()
    cp.train(X_train, y_train)

    pred     = cp.predict(X_test)
    pred_enc = cp.predict_encoded(X_test)

    for p, p_ in zip(pred, pred_enc):
        for c, c_ in zip(p, p_):
            for i in range(len(cp.labels)):
                if c_ & 0b1 << i > 0:
                    assert cp.labels[i] in c
