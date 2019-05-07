from .test_rs import sum_vec

def test_sum_rust(benchmark):
    x = benchmark(sum_vec, [1, 2, 3, 4])
    assert x == 10.0

def test_sum_python(benchmark):
    x = benchmark(sum, [1, 2, 3, 4])
    assert x == 10
