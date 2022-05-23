# -*- coding: utf-8 -*-
"""
Created on 2020/01/02
@author: Hao Sun

Created on Sat Feb 18 16:21:13 2017
@author: Xiangyong Cao
This code is modified based on https://github.com/KGPML/Hyperspectral
"""

import scipy.io
import numpy as np
from random import shuffle
import random
import scipy.ndimage
from skimage.util import pad
import os
import time
import pandas as pd
import math
from utils_houston import patch_size, convertToOneHot, Prepare_data, Database


flag_augment = False   # true if adopt data-augmentation strategy

start_time = time.time()
print(Database)
## Load data
Database_path = 'Data/' + Database 
DATA_PATH = os.path.join(os.getcwd(),Database_path)
Data = scipy.io.loadmat(os.path.join(DATA_PATH, 'DFC2013.mat'))['hsi']
Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'DFC2013.mat'))['gt_all']
Label_train = scipy.io.loadmat(os.path.join(DATA_PATH, 'DFC2013.mat'))['gt_train']
Label_test = scipy.io.loadmat(os.path.join(DATA_PATH, 'DFC2013.mat'))['gt_test']

## Some constant parameters
Height, Width, Band = Data.shape[0], Data.shape[1], Data.shape[2]
Num_Classes = len(np.unique(Label))-1     


## Scale the HSI Data between [0,1]
Data = Data.astype(float)
print('--------------------normalize HSI -------------------------------')
for band in range(Band):
    Data[:,:,band] = (Data[:,:,band]-np.min(Data[:,:,band]))/(np.max(Data[:,:,band])-np.min(Data[:,:,band]))

## padding the data beforehand
Height, Width, Band = Data.shape[0], Data.shape[1], Data.shape[2]
Data_Padding = np.zeros((Height+int(patch_size-1),Width+int(patch_size-1),Band))
for band in range(Band):
    Data_Padding[:,:,band] = pad(Data[:,:,band],int((patch_size-1)/2),'symmetric')
    
def Patch(height_index,width_index):
    """ function to extract patches from the orignal data """
    transpose_array = np.transpose(Data_Padding,(2,0,1))
    height_slice = slice(height_index, height_index + patch_size)
    width_slice = slice(width_index, width_index + patch_size)
    patch = transpose_array[:,height_slice, width_slice]
    return np.array(patch)   

## Collect patches for each class based on gt_all
Classes, Classes_Index = [], []
for k in range(Num_Classes):
    Classes.append([])
    Classes_Index.append([])

#All_Patches, All_Labels = [],[]   
for j in range(0,Width):
    for i in range(0,Height):
        curr_inp = Patch(i,j)
        curr_ind = j * Height + i
        curr_tar = Label[i,j]
        #All_Patches.append(curr_inp)
        #All_Labels.append(curr_tar)
        #Ignore patches with unknown landcover type for the central pixel
        if(curr_tar!=0): 
            Classes[curr_tar-1].append(curr_inp) 
            Classes_Index[curr_tar-1].append(curr_ind)
#All_data = {}
#All_data['patch'] = All_Patches
#All_data['labels'] = All_Labels
#scipy.io.savemat(os.path.join(DATA_PATH, 'All_data'+'_'+str(patch_size)+'.mat'),All_data)

## Collect patches for each class based on gt_train 
Classes_train,Classes_Index_train = [],[]
for k in range(Num_Classes):
    Classes_train.append([])
    Classes_Index_train.append([])

for j in range(0,Width):
    for i in range(0,Height):
        curr_inp = Patch(i,j)
        curr_ind = j * Height + i
        curr_tar = Label_train[i,j]
        #Ignore patches with unknown landcover type for the central pixel
        if(curr_tar!=0): 
            Classes_train[curr_tar-1].append(curr_inp) 
            Classes_Index_train[curr_tar-1].append(curr_ind)

## Collect patches for each class based on gt_test 
Classes_test,Classes_Index_test = [],[]
for k in range(Num_Classes):
    Classes_test.append([])
    Classes_Index_test.append([])

for j in range(0,Width):
    for i in range(0,Height):
        curr_inp = Patch(i,j)
        curr_ind = j * Height + i
        curr_tar = Label_test[i,j]
        #Ignore patches with unknown landcover type for the central pixel
        if(curr_tar!=0): 
            Classes_test[curr_tar-1].append(curr_inp) 
            Classes_Index_test[curr_tar-1].append(curr_ind)

#  num of training and test pixels
Num_Each_Class, Num_Train_Each_Class, Num_Test_Each_Class = [], [], []
for k in range(Num_Classes):
    Num_Each_Class.append(len(Classes[k]))
    Num_Train_Each_Class.append(len(Classes_train[k]))
    Num_Test_Each_Class.append(len(Classes_test[k]))

Train_Patch, Train_Label, TrainIndex, Test_Patch, Test_Label, TestIndex = [],[],[],[], [], []

for k in range(Num_Classes): 
    train_patch = Classes_train[k]
    test_patch = Classes_test[k]
    train_index = Classes_Index_train[k] 
    test_index = Classes_Index_test[k]

    TestIndex.extend(test_index)
    TrainIndex.extend(train_index)

    Train_Patch.append(train_patch)   
    Test_Patch.extend(test_patch)   

    #Train_Label.append(np.full(Num_Train_Each_Class[k], k, dtype=int))
    Test_Label.extend(np.full(Num_Test_Each_Class[k], k, dtype=int))

for k in range(Num_Classes):
    Train_Label.append([k]*Num_Train_Each_Class[k])

Resample_Num_Count = Num_Train_Each_Class
######### crop train cubes based on gt_train and gt_test   
# Augment the data with random flipped and rotated patches
fixed_Train_Patch = Train_Patch    
if flag_augment:
    Resample_Num_Count = []
    #times = 5    # can be tuned
    for k in range(Num_Classes):
        #for l in range(times*Num_Train_Each_Class[k]):
        for l in range(Num_Train_Each_Class[k]):
            current_patch = Train_Patch[k][l]          #C*H*W
            current_patch_trans = np.transpose(current_patch,[1,2,0]) #C*H*W -> H*W*C

            # flip up->down
            flipped_patch = []
            flipped_patch = np.flipud(current_patch_trans)
            flipped_patch = np.transpose(flipped_patch,[2,0,1]) #  H*W*C  ->  C*H*W
            #print('flipped_patch.shape', flipped_patch.shape) 
            Train_Patch[k].append(flipped_patch)
            Train_Label[k].append(k)

            # flip left->right
            flipped_patch = []
            flipped_patch = np.fliplr(current_patch_trans) 
            flipped_patch = np.transpose(flipped_patch,[2,0,1]) #  H*W*C  ->  C*H*W
            Train_Patch[k].append(flipped_patch)
            Train_Label[k].append(k)

            # rotate patch 90 counterclockwise
            flipped_patch = []
            flipped_patch = np.rot90(current_patch_trans, k=1)  
            flipped_patch = np.transpose(flipped_patch,[2,0,1]) #  H*W*C  ->  C*H*W
            Train_Patch[k].append(flipped_patch)
            Train_Label[k].append(k)

            # rotate patch 180 counterclockwise
            flipped_patch = []
            flipped_patch = np.rot90(current_patch_trans, k=2)  
            flipped_patch = np.transpose(flipped_patch,[2,0,1]) #  H*W*C  ->  C*H*W
            Train_Patch[k].append(flipped_patch)
            Train_Label[k].append(k)

            # rotate patch 270 counterclockwise
            flipped_patch = []
            flipped_patch = np.rot90(current_patch_trans, k=3)  
            flipped_patch = np.transpose(flipped_patch,[2,0,1]) #  H*W*C  ->  C*H*W
            Train_Patch[k].append(flipped_patch)
            Train_Label[k].append(k)
   
            #if(len(Train_Patch[k])<times*Num_Train_Each_Class[k]):   
            #    #num = random.randint(0,2)
            #    j = random.randint(0,Num_Train_Each_Class[k]-1)
            #    #noise = fixed_Train_Patch[k][j] 
            #    noise = fixed_Train_Patch[k][j] + np.random.normal(0,0.001,size = fixed_Train_Patch[k][l].shape) 
            #    #if num == 0 :
            #    #    #Flip patch up-down
            #    #    flipped_patch = np.flipud(noise) 
            #    #if num == 1 :
            #    #    #Flip patch left-right
            #    #    flipped_patch = np.fliplr(noise) 
            #    #if num == 2 :
            #    #    #Rotate patch by a random angle
            #    #    no = random.randrange(-180,180,90)
            #    #    flipped_patch = scipy.ndimage.interpolation.rotate(noise, no,axes=(1, 0), 
            #    #        reshape=False, output=None, order=3, mode='constant', cval=0.0, prefilter=False)
            #    Train_Patch[k].append(noise)
            #    Train_Label[k].append(k)

        Resample_Num_Count.append(len(Train_Patch[k]))
                                
    OS_Aug_Num_Training_Each = []             
    for k in range(Num_Classes):
        OS_Aug_Num_Training_Each.append(len(Train_Label[k])) 

if flag_augment == False:
    OS_Aug_Num_Training_Each = Resample_Num_Count
            
# release list to elements             
Temp1,Temp2 = [],[]
for k in range(Num_Classes):
    Temp1.extend(Train_Patch[k])
    Temp2.extend(Train_Label[k])
Train_Patch = Temp1
Train_Label = Temp2

Train_Patch = np.array(Train_Patch)

# Convert the labels to One-Hot vector
Train_Label = np.array(Train_Label)
Test_Label = np.array(Test_Label)
Train_Label = convertToOneHot(Train_Label,num_classes=Num_Classes)
Test_Label = convertToOneHot(Test_Label,num_classes=Num_Classes)
                      
## Save the patches in segments
# Train Data
train_dict = {}
#file_name = 'Train_'+str(patch_size)+'.mat'
#train_dict["train_patch"] = Train_Patch
#train_dict["train_labels"] = Train_Label
#scipy.io.savemat(os.path.join(DATA_PATH, file_name),train_dict)
num_train = len(Train_Patch)
num_train_file = 10
num_each_file = int(num_train / num_train_file)
res_num = num_train - num_train_file * num_each_file
Num_Each_File = num_each_file * np.ones((1,num_train_file),dtype=int)
Num_Each_File = Num_Each_File[0]
Num_Each_File[num_train_file-1] = Num_Each_File[num_train_file-1] + res_num
start = 0
for i in range(num_train_file):
    file_name = 'Train_'+str(patch_size)+'_'+str(i+1)+'.mat'
    train_dict["train_patch"] = Train_Patch[start:start+Num_Each_File[i]]
    train_dict["train_labels"] = Train_Label[start:start+Num_Each_File[i],:]
    scipy.io.savemat(os.path.join(DATA_PATH, file_name),train_dict)
    start = start + Num_Each_File[i]
    
# Test Data
test_dict = {}
#file_name = 'Test_'+str(patch_size)+'.mat'
#test_dict["test_patch"] = Test_Patch
#test_dict["test_labels"] = Test_Label
#scipy.io.savemat(os.path.join(DATA_PATH, file_name),test_dict)
num_test = len(Test_Patch)
num_test_file = 10
num_each_file = int(num_test / num_test_file)
res_num = num_test - num_test_file * num_each_file
Num_Each_File = num_each_file * np.ones((1,num_test_file),dtype=int)
Num_Each_File = Num_Each_File[0]
Num_Each_File[num_test_file-1] = Num_Each_File[num_test_file-1] + res_num
start = 0
for i in range(num_test_file):
    file_name = 'Test_'+str(patch_size)+'_'+str(i+1)+'.mat'
    test_dict["test_patch"] = Test_Patch[start:start+Num_Each_File[i]]
    test_dict["test_labels"] = Test_Label[start:start+Num_Each_File[i],:]
    scipy.io.savemat(os.path.join(DATA_PATH, file_name),test_dict)
    start += Num_Each_File[i]



train_ind = {}
train_ind['TrainIndex'] = TrainIndex
scipy.io.savemat(os.path.join(DATA_PATH, 'TrainIndex'+'_'+str(patch_size)+'.mat'),train_ind)

test_ind = {}
test_ind['TestIndex'] = TestIndex
scipy.io.savemat(os.path.join(DATA_PATH, 'TestIndex'+'_'+str(patch_size)+'.mat'),test_ind)




Training_data, Test_data = Prepare_data()
  
print('Training Data:')
print(Training_data['train_patch'].shape)
print('Test Data:')
print(Test_data['test_patch'].shape)

# Data Summary
df = pd.DataFrame(np.random.randn(Num_Classes, 4),
                  columns=['Total', 'Training', 'OS&Aug', 'Testing'])
df['Total'] = Num_Each_Class
df['Training'] = Num_Train_Each_Class
df['OS&Aug'] = OS_Aug_Num_Training_Each
df['Testing'] = Num_Test_Each_Class
print("=======================================================================")
print("Data Summary")
print("=======================================================================")
print('The size of the original HSI data is (%d,%d,%d)'%(Height,Width,Band))
print('The size of Training data is (%d)'%(num_train))
print('The size of Test data is (%d)'%(num_test))
print('The size of each sample is (%d,%d,%d)'%(Band,patch_size,patch_size))
print('-----------------------------------------------------------------------')
print("The Data Division is")
print(df)
duration_time = time.time() - start_time
print("=======================================================================")
print('Data Preparation is Completed! (It takes %.5f seconds)'%(duration_time))
print("=======================================================================")

