import math

def zero_one(pred, true):
    if pred == true: return 0.0
    return 1.0

def squared(pred, true):
    return (pred - true) ** 2

def absolute(pred, true):
    return math.abs(pred - true)

def absolute_percentage(pred, true):
    return math.abs( (pred - true) / true )
