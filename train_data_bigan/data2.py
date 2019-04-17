import logging
import numpy as np
import pandas as pd
import bigan.network2 as network
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
characters=network.characters
times=network.times
logger = logging.getLogger(__name__)
"""
取样规则变化，训练集和测试集各取50% ，训练集里只取正常，测试集里n:a =4:1
"""
def get_train(*args):
    """Get training dataset """
    return _get_adapted_dataset("train")

def get_test(*args):
    """Get testing dataset """
    return _get_adapted_dataset("test")

def get_shape_input():
    """Get shape of the dataset"""
    return (None, characters,times)

def get_shape_label():
    """Get shape of the labels """
    return (None,)
def _get_dataset():
    n_data = np.loadtxt("train_data_bigan/n",delimiter=' ')
    a_data = np.loadtxt("train_data_bigan/a",delimiter=' ')
    n_batch_num=int(n_data.shape[0]/characters)
    a_batch_num=int(a_data.shape[0]/characters)
    n_data=np.reshape(n_data,[-1,characters,times])
    a_data=np.reshape(a_data,[-1,characters,times])
    n_data_label=np.zeros((n_batch_num),dtype=np.int)
    a_data_label=np.ones((a_batch_num),dtype=np.int)

    data=np.concatenate([n_data,a_data],axis=0)
    data_label=np.concatenate([n_data_label,a_data_label],axis=0)

    rng = np.random.RandomState(42)
    inds = rng.permutation(data.shape[0])
    data=data[inds]
    data_label=data_label[inds]

    fen=int(data.shape[0]/2)
    x_train=data[:fen]
    x_test=data[fen:]

    y_train=data_label[:fen]
    y_test=data_label[fen:]

    x_train=x_train[y_train!=1]
    y_train=y_train[y_train!=1]

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    return  dataset

def _get_adapted_dataset(split):
    """ Gets the adapted dataset for the experiments
    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    dataset = _get_dataset()
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if split != 'train':
        dataset[key_img], dataset[key_lbl] = _adapt(dataset[key_img],
                                                    dataset[key_lbl])

    return (dataset[key_img], dataset[key_lbl])

def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)

def _adapt(x, y, rho=0.2):
    """Adapt the ratio of normal/anomalous data"""

    # Normal data: label =0, anomalous data: label =1   所有的正常加上异常里出来一部分

    rng = np.random.RandomState(42) # seed shuffling

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_test = inliersx.shape[0]
    out_size_test = int(size_test*rho/(1-rho)) #n:a 4:1

    outestx = outliersx[:out_size_test]
    outesty = outliersy[:out_size_test]

    testx = np.concatenate((inliersx,outestx), axis=0)
    testy = np.concatenate((inliersy,outesty), axis=0)

    size_test = testx.shape[0]
    inds = rng.permutation(size_test)
    testx, testy = testx[inds], testy[inds]

    return testx, testy