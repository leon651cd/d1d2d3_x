import logging
import numpy as np
import pandas as pd
import bigan.network3 as network
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
characters=network.characters
times=network.times
logger = logging.getLogger(__name__)
"""
适配加入D2
train_data增加异常数据
train_data里x_train,y_train用于训练D1
x_train_a,y_train_a用于训练D2
"""
def get_train(*args):
    """Get training dataset """
    return _get_adapted_dataset("train")

def get_train_a(*args):
    return _get_adapted_dataset("train_a")

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
    n_data = np.loadtxt("train_data_bigan/n", delimiter=' ')
    a_data = np.loadtxt("train_data_bigan/a", delimiter=' ')


    n_data=np.reshape(n_data,[-1,characters,times])
    a_data=np.reshape(a_data,[-1,characters,times])

    trian_ratio=0.6  #训练集所占的比例

    rng = np.random.RandomState(42)

    inds = rng.permutation(n_data.shape[0])
    n_data=n_data[inds]
    fen_n=int(n_data.shape[0]*trian_ratio)
    n_data_train=n_data[:fen_n]
    n_data_test=n_data[fen_n:]
    n_data_train_label=np.zeros((n_data_train.shape[0]),dtype=np.int)
    n_data_test_label=np.zeros((n_data_test.shape[0]),dtype=np.int)

    inds = rng.permutation(a_data.shape[0])
    a_data=a_data[inds]
    fen_a=int(a_data.shape[0]*trian_ratio)
    a_data_train=a_data[:fen_a]
    a_data_test=a_data[fen_a:]
    a_data_train_label = np.ones((a_data_train.shape[0]), dtype=np.int)
    a_data_test_label = np.ones((a_data_test.shape[0]), dtype=np.int)

    data_test=np.concatenate([n_data_test,a_data_test],axis=0)
    data_test_label=np.concatenate([n_data_test_label,a_data_test_label],axis=0)

    x_train=n_data_train
    y_train=n_data_train_label

    x_train_a=a_data_train
    y_train_a=a_data_train_label

    x_test=data_test
    y_test=data_test_label

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    dataset['x_train_a']=x_train_a.astype(np.float32)
    dataset['y_train_a']=y_train_a.astype(np.float32)
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

    if split == 'test1':# debuging!!!!!!!!!!!!现在都没调整!!!!!
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
