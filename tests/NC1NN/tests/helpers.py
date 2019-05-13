def vec_cmp(a1, a2):
    for i in range(a1.shape[0]):
        assert a1[i] == a2[i]
