import numpy as np

class LabelMap:
    def __init__(self):
        self.map         = {}
        self.reverse_map = {}
        self.label_count = 0

    def transform(self, label):
        label = as_argmax(label)

        if label in self.map:
            return self.map[label]
        else:
            self.map[label] = self.label_count
            self.reverse_map[self.label_count] = label
            self.label_count += 1
            return self.label_count - 1

    def reverse(self, idx):
        return self.reverse_map[as_argmax(idx)]

    def __iter__(self):
        return range(self.label_count).__iter__()

    def __len__(self):
        return self.label_count

def append(X, y, X_, y_):
    if len(y.shape) > 1:
        return np.vstack((X, X_)), np.vstack((y, y_))
    else:
        return np.vstack((X, X_)), np.append(y, y_)

def format(X, y = None, label_map = None):
    X = __list_to_ndarray(X)
    X = __reshape_if_vector(X)

    if y is not None:
        y = __reshape_if_scalar(y)

        if label_map != None:
            y_transformed = np.array([
                label_map.transform(y_) for y_ in y
            ])
            if len(y.shape) == 2:
                # neural nets have a 2d output and not 1d
                # so do not change y, but keep label_map
                # for internal work
                return X, y
            else:
                return X, y_transformed
        return X, y
    return X

def as_argmax(y):
    if type(y) is np.ndarray:
        return np.argmax(y)
    return y

def __list_to_ndarray(z):
    return np.array(z) if type(z) is list else z

def __reshape_if_vector(X):
    return X.reshape(1, X.shape[0]) \
        if len(X.shape) == 1 else X

def __reshape_if_scalar(y):
    return np.array([y]) if type(y) is not np.ndarray \
        else y
