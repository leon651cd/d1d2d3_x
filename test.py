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
train 和 test的划分固定

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

def list_shuffle(x,inds):
    inds=inds.tolist()
    x_new=[]
    for i in inds:
        x_new += (x[i * 10:(i + 1) * 10])
    return x_new

def slot_index(inds):
    inds = inds.tolist()
    new_inds=[]
    for i in inds:
        for j in range(i*10 , (i+1)*10):
            new_inds.append(int(j))
    return np.array(new_inds)

def slot_index1(inds):
    inds = inds.tolist()
    loc=[]
    for i in range(len(inds)):
        if(inds[i]==1):
            loc.append(i)
    loc=np.array(loc)
    return slot_index(loc)

# a=[1,2,3,4,5]
# a=np.array(a)
# inds=slot_index(a)
# print(inds)
# x_new=list_shuffle(a,inds)
# print(x_new)
def train_a_ip(src,des):
    src=src.tolist()
    src_ip={}
    des_ip={}
    for i in src:
        if i not in src_ip:
            src_ip[i]=1
        else:
            src_ip[i]+=1
    for i in des:
        ips=i.strip().split(",")
        for j in ips:
            if( j=='0'):
                break
            if j not in des_ip :
                des_ip[j]=1
            else:
                des_ip[j]+=1
    return src_ip,des_ip

def list_delete(list,inds):
    inds=inds.tolist()
    new_list=[]
    for i in inds:
        if i < len(list):
            new_list.append(list[i])
    return  new_list

def list_select(list,inds):
    inds=inds.tolist()
    new_list = []
    for i in inds:
        if i < len(list):
            new_list.append(list[i])
    # for i in range(len(list)):
    #     if i in inds:
    #         new_list.append(list[i])
    return new_list

# a=[4,6,3,8,4,2,7]
# n=np.array([0,5,3])
# print(list_delete(a,n))
def _get_dataset(ratio):

    file_name='98_pca_bigan_fix/'
    a_data_train=np.loadtxt(file_name+"a_data_train", delimiter=' ')
    a_ip_train=np.loadtxt(file_name+"a_ip_train", delimiter=' ',dtype=str)
    a_label_train=np.loadtxt(file_name+"a_label_train", delimiter=' ')
    a_attack_train=np.loadtxt(file_name+"a_attack_train", delimiter=' ')
    a_slot_attack_train=open(file_name + "a_slot_attack_train").readlines()
    a_slot_desip_train=open(file_name + "a_slot_desip_train").readlines()
    a_slot_all_desip_train = open(file_name + "a_slot_all_desip_train").readlines()
    a_data_train_label=np.ones((a_data_train.shape[0]),dtype=np.int)

    n_data_train = np.loadtxt(file_name + "n_data_train", delimiter=' ')
    n_ip_train = np.loadtxt(file_name + "n_ip_train", delimiter=' ', dtype=str)
    n_label_train = np.loadtxt(file_name + "n_label_train", delimiter=' ')
    # a_attack_train = np.loadtxt(file_name + "a_attack_train", delimiter=' ')
    n_slot_attack_train = open(file_name + "n_slot_attack_train").readlines()
    n_slot_desip_train = open(file_name + "n_slot_desip_train").readlines()
    n_slot_all_desip_train = open(file_name + "n_slot_all_desip_train").readlines()
    n_data_train_label = np.zeros((n_data_train.shape[0]), dtype=np.int)

    data_test=np.loadtxt(file_name + "data_test", delimiter=' ')
    data_test_label=np.loadtxt(file_name + "data_test_label", delimiter=' ')
    ip_test=np.loadtxt(file_name + "ip_test", delimiter=' ', dtype=str)
    label_test=np.loadtxt(file_name + "label_test", delimiter=' ')
    slot_attack_test = open(file_name + "slot_attack_test").readlines()
    slot_desip_test = open(file_name + "slot_desip_test").readlines()
    slot_all_desip_test = open(file_name + "slot_all_desip_test").readlines()

    n_data_train = np.reshape(n_data_train, [-1, 1, characters])
    a_data_train = np.reshape(a_data_train, [-1, 1, characters])

    if (ratio == 2):
        # #将某种异常的全部标签去除
        inds=1*(a_attack_train == 1)
        slot_inds=slot_index1(inds)
        # print(slot_inds)
        # print(slot_inds.shape)
        a_data_train_qu = a_data_train[a_attack_train == 1]
        a_data_train_label_qu=a_data_train_label[a_attack_train == 1]
        a_label_train_qu = a_label_train[a_attack_train == 1]
        a_ip_train_qu=a_ip_train[a_attack_train == 1]
        a_slot_attack_train_qu=list_select(a_slot_attack_train,slot_inds)
        a_slot_desip_train_qu=list_select(a_slot_desip_train,slot_inds)
        a_slot_all_desip_train_qu = list_select(a_slot_all_desip_train,slot_inds)

        indn=1*(a_attack_train == 0)
        slot_indn=slot_index1(indn)
        a_data_train = a_data_train[a_attack_train == 0]
        a_data_train_label=a_data_train_label[a_attack_train == 0]
        a_label_train = a_label_train[a_attack_train == 0]
        a_ip_train = a_ip_train[a_attack_train == 0]
        a_slot_attack_train = list_select(a_slot_attack_train,slot_indn)
        a_slot_desip_train = list_select(a_slot_desip_train,slot_indn)
        a_slot_all_desip_train = list_select(a_slot_all_desip_train,slot_indn)

        n_data_train=np.concatenate([n_data_train,a_data_train_qu],axis=0)
        n_data_train_label=np.concatenate([n_data_train_label,a_data_train_label_qu],axis=0)
        n_label_train = np.concatenate([n_label_train, a_label_train_qu], axis=0)
        n_ip_train = np.concatenate([n_ip_train, a_ip_train_qu], axis=0)
        n_slot_attack_train=n_slot_attack_train+a_slot_attack_train_qu
        n_slot_desip_train = n_slot_desip_train + a_slot_desip_train_qu
        n_slot_all_desip_train = n_slot_all_desip_train + a_slot_all_desip_train_qu

    # else:
        # num = int(a_data_train.shape[0] * ratio)
        # p_inds = np.random.choice(a_data_train.shape[0], num, replace=False)
        # slot_p_inds=slot_index(p_inds)
        # a_data_train_qu = a_data_train[p_inds]
        # a_data_train_label_qu=a_data_train_label[p_inds]
        # a_label_train_qu = a_label_train[p_inds]
        # a_ip_train_qu = a_ip_train[p_inds]
        # a_slot_attack_train_qu = list_select(a_slot_attack_train,slot_p_inds)
        # a_slot_desip_train_qu = list_select(a_slot_desip_train,slot_p_inds)
        # a_slot_all_desip_train_qu = list_select(a_slot_all_desip_train,slot_p_inds)
        #
        # a_data_train = np.delete(a_data_train, p_inds, axis=0)
        # a_data_train_label=np.delete(a_data_train_label,p_inds,axis=0)
        # a_label_train = np.delete(a_label_train, p_inds, axis=0)
        # a_ip_train = np.delete(a_ip_train, p_inds, axis=0)
        # a_slot_attack_train = list_delete(a_slot_attack_train,slot_p_inds)
        # a_slot_desip_train = list_delete(a_slot_desip_train,slot_p_inds)
        # a_slot_all_desip_train = list_delete(a_slot_all_desip_train,slot_p_inds)
        #
        # n_data_train = np.concatenate([n_data_train, a_data_train_qu], axis=0)
        # n_data_train_label=np.concatenate([n_data_train_label, a_data_train_label_qu], axis=0)
        # n_label_train = np.concatenate([n_label_train, a_label_train_qu], axis=0)
        # n_ip_train = np.concatenate([n_ip_train, a_ip_train_qu], axis=0)
        # n_slot_attack_train = n_slot_attack_train + a_slot_attack_train_qu
        # n_slot_desip_train = n_slot_desip_train + a_slot_desip_train_qu
        # n_slot_all_desip_train = n_slot_all_desip_train + a_slot_all_desip_train_qu

    data_test=np.reshape(data_test, [-1, 1, characters])

    # print(a_data_train.shape)
    # print(n_data_train.shape)
    # print(data_test.shape)
    print('desip！!！!！!！')
    print(a_slot_desip_train)

    return a_data_train,a_data_train_label, a_ip_train, a_label_train, a_slot_attack_train, a_slot_desip_train, a_slot_all_desip_train, \
           n_data_train,n_data_train_label, n_ip_train, n_label_train, n_slot_attack_train, n_slot_desip_train, n_slot_all_desip_train, \
           data_test, ip_test, label_test, slot_attack_test, slot_desip_test, slot_all_desip_test, data_test_label





    # src_ip,des_ip=train_a_ip(a_ip_train,a_slot_desip_train)



_get_dataset(0)
# print(len(a['slot_attack']))
# print(a["slot_attack"])

def _get_adapted_dataset(ratio):
    """ Gets the adapted dataset for the experiments
    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    dataset, src_ip, des_ip = _get_dataset(ratio)
    #trainx, trainy,trainx_a,trainy_a,testx,testy,test_label,test_ip,a_label_train= data._get_adapted_dataset(ratio)
    return dataset['x_train'], dataset["y_train"], dataset['x_train_a'], dataset['y_train_a'], dataset['x_test'], \
           dataset['y_test'], dataset['label'], dataset['ip'], dataset['a_label_train'], dataset['slot_attack'], \
           dataset['slot_desip'],dataset['slot_all_desip'],src_ip,des_ip
