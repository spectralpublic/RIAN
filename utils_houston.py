# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 16:21:13 2017
@author: Xiangyong Cao
This code is modified based on https://github.com/KGPML/Hyperspectral
"""

import numpy as np
import os
import scipy.io

Database = 'Houston'   # Indian_pine      Pavia_university
patch_size = 9   # can be tuned

num_band = 144    # paviaU 103   indian_pine 220 
Height = 349   # paviaU 610   indian_pine 145
Width = 1905    # paviaU 340  indian_pine 145
Band = num_band
num_classes = 15  # paviaU 9   indian_pine 16
n_classes = num_classes


num_train_file = 10
num_test_file = 10
Database_path = 'Data/' + Database 
DATA_PATH = os.path.join(os.getcwd(),Database_path)


#All_data = scipy.io.loadmat(os.path.join(DATA_PATH, 'AllData.mat'))
#TrainIndex = scipy.io.loadmat(os.path.join(DATA_PATH, 'TrainIndex.mat'))['TrainIndex']
#TestIndex = scipy.io.loadmat(os.path.join(DATA_PATH, 'TestIndex.mat'))['TestIndex']

def add_DataSet(first,second,data_type):
    if data_type == 'train':
        temp_image = np.concatenate((first['train_patch'],second['train_patch']),axis=0)
        temp_labels = np.concatenate((first['train_labels'],second['train_labels']),axis=0)
        Combined_data = {}
        Combined_data['train_patch'] = temp_image
        Combined_data['train_labels'] = temp_labels
    if data_type == 'test':
        temp_image = np.concatenate((first['test_patch'],second['test_patch']),axis=0)
        temp_labels = np.concatenate((first['test_labels'],second['test_labels']),axis=0)
        Combined_data = {}
        Combined_data['test_patch'] = temp_image
        Combined_data['test_labels'] = temp_labels
    return Combined_data

def Prepare_data():
    """ functions to prepare Training and Testing data"""
    for i in range(num_train_file):
        file_name = 'Train_'+str(patch_size)+'_'+str(i+1)+'.mat'
        data_sets = scipy.io.loadmat(os.path.join(DATA_PATH, file_name))
        if(i==0):
            Training_data = data_sets
            continue
        else:
            Training_data = add_DataSet(Training_data,data_sets,'train')
            
    for i in range(num_test_file):
        file_name = 'Test_'+str(patch_size)+'_'+str(i+1)+'.mat'
        data_sets = scipy.io.loadmat(os.path.join(DATA_PATH, file_name))
        if(i==0):
            Test_data = data_sets
            continue
        else:
            Test_data = add_DataSet(Test_data,data_sets,'test')
    return Training_data, Test_data

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

