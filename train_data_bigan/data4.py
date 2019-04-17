import logging
import numpy as np
import pandas as pd
import bigan.network7 as network
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
characters=network.characters
times=network.times
latent_dim=network.latent_dim
logger = logging.getLogger(__name__)
"""
适配加入D2 pca
train_data增加异常数据
train_data里x_train,y_train用于训练D1
x_train_a,y_train_a用于训练D2
test数据里面增加label、IP等
"""
def get_train(ratio):
    """Get training dataset """
    return _get_adapted_dataset("train",ratio)

def get_train_a(ratio):
    return _get_adapted_dataset("train_a",ratio)

def get_a_label_train(ratio):
    return _get_adapted_dataset("a_label_train",ratio)

def get_test(ratio):
    """Get testing dataset """
    return _get_adapted_dataset("test",ratio)

def get_test_label(ratio):
    return _get_adapted_dataset("label",ratio)

def get_test_ip(ratio):
    return _get_adapted_dataset("ip",ratio)

def get_shape_input():
    """Get shape of the dataset"""
    return (None, 1,characters)

def get_shape_labels():
    """Get shape of the dataset"""
    return (None, characters)

def get_shape_label():
    """Get shape of the labels """
    return (None,)
def _get_dataset(ratio):
    n_data = np.loadtxt("98_pca_bigan/n_pca_transpose", delimiter=' ')
    a_data = np.loadtxt("98_pca_bigan/a_pca_transpose", delimiter=' ')

    n_ip = np.loadtxt("98_pca_bigan/n_ip", delimiter=' ',dtype=str)
    a_ip = np.loadtxt("98_pca_bigan/a_ip", delimiter=' ',dtype=str)

    n_label = np.loadtxt("98_pca_bigan/n_label", delimiter=' ')
    a_label = np.loadtxt("98_pca_bigan/a_label", delimiter=' ')

    a_attack =np.loadtxt("98_pca_bigan/a_attack",dtype=int) #获取是否有某类异常的全部标签

    n_data=np.reshape(n_data,[-1,1,characters])
    a_data=np.reshape(a_data,[-1,1,characters])

    trian_ratio=0.6  #训练集所占的比例

    rng = np.random.RandomState(42)

    inds = rng.permutation(n_data.shape[0])
    n_data=n_data[inds]
    n_label=n_label[inds]
    n_ip=n_ip[inds]
    fen_n=int(n_data.shape[0]*trian_ratio)
    n_data_train=n_data[:fen_n]
    n_data_test=n_data[fen_n:]
    n_label_train=n_label[:fen_n]
    n_label_test=n_label[fen_n:]
    n_ip_train=n_ip[:fen_n]
    n_ip_test=n_ip[fen_n:]
    n_data_train_label=np.zeros((n_data_train.shape[0]),dtype=np.int)
    n_data_test_label=np.zeros((n_data_test.shape[0]),dtype=np.int)

    inds = rng.permutation(a_data.shape[0])
    a_data=a_data[inds]
    a_label=a_label[inds]
    a_ip=a_ip[inds]
    a_attack=a_attack[inds]
    fen_a=int(a_data.shape[0]*trian_ratio)
    a_data_train=a_data[:fen_a]
    a_data_test=a_data[fen_a:]
    a_label_train=a_label[:fen_a]

    # print("a_label_train")
    # print(a_label_train.shape)

    a_label_test=a_label[fen_a:]
    a_ip_train=a_ip[:fen_a]
    a_ip_test=a_ip[fen_a:]
    a_attack_train=a_attack[:fen_a]
    a_attack_test=a_attack[fen_a:]
    a_data_train_label = np.ones((a_data_train.shape[0]), dtype=np.int)
    a_data_test_label = np.ones((a_data_test.shape[0]), dtype=np.int)

    data_test=np.concatenate([n_data_test,a_data_test],axis=0)
    label_test=np.concatenate([n_label_test,a_label_test],axis=0)
    ip_test=np.concatenate([n_ip_test,a_ip_test],axis=0)
    data_test_label=np.concatenate([n_data_test_label,a_data_test_label],axis=0)

    #在这里随机的抽取一定比例的异常到正常中,dont forget put a_label_train away
    x_train=n_data_train
    y_train=n_data_train_label
    x_train_a=a_data_train
    y_train_a=a_data_train_label


    #随机抽10%的异常到正常中
    if(ratio==2):
        # #将某种异常的全部标签去除
        x_train_a_qu = x_train_a[a_attack_train == 1]
        a_label_train_qu = a_label_train[a_attack_train == 1]
        x_train_a = x_train_a[a_attack_train == 0]
        a_label_train = a_label_train[a_attack_train == 0]
        x_train = np.concatenate([x_train, x_train_a_qu], axis=0)
        n_label_train = np.concatenate([n_label_train, a_label_train_qu], axis=0)

    else:
        num = int(x_train_a.shape[0] * ratio)
        p_inds = np.random.choice(x_train_a.shape[0], num, replace=False)
        x_train_a_ratio = x_train_a[p_inds]
        a_label_train_ratio = a_label_train[p_inds]

        x_train_a = np.delete(x_train_a, p_inds, axis=0)
        a_label_train = np.delete(a_label_train, p_inds, axis=0)

        x_train = np.concatenate([x_train, x_train_a_ratio], axis=0)
        n_label_train = np.concatenate([n_label_train, a_label_train_ratio], axis=0)







    print("normal num:")
    print(x_train.shape[0])
    print("anomaly num:")
    print(x_train_a.shape[0])



    x_test=data_test
    label=label_test
    ip=ip_test
    y_test=data_test_label

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)
    dataset['x_train_a']=x_train_a.astype(np.float32)
    dataset['a_label_train'] = a_label_train
    dataset['y_train_a']=y_train_a.astype(np.float32)
    dataset['label']=label
    dataset['ip']=ip


    return  dataset

def _get_adapted_dataset(split,ratio):
    """ Gets the adapted dataset for the experiments
    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    dataset = _get_dataset(ratio)
    key_img = 'x_' + split
    key_lbl = 'y_' + split
    if(split=='label' or split =='ip' or split=='a_label_train'):
        return dataset[split]

    return (dataset[key_img], dataset[key_lbl])

