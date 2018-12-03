# coding:utf-8
import pickle
import numpy as np

def to_onehot(y):
    return np.identity(10)[y].reshape((y.shape[0], 10))

def load_data(file_name):
    names = ['x', 'y', 'tx', 'ty']
    data = []
    for name in names:
        with open('data/{}.{}'.format(file_name, name), 'rb') as fp:
            data.append(pickle.load(fp))

    return tuple(data)

def next_batch(x, y, batch_size):
    idx = np.arange(0 , len(x))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    x_shuffle = [x[i] for i in idx]
    y_shuffle = [y[i] for i in idx]
    return np.array(x_shuffle, dtype = np.float32), np.asarray(y_shuffle, dtype = np.int32)
