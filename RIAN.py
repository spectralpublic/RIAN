# -*- coding: utf-8 -*-
"""
The paper of this code is currently under review. If you find the code and dataset useful in your research, please consider citing our paper.

X. Zheng, H. Sun, X. Lu, and W. Xie, “Rotation-Invariant Attention Network for Hyperspectral Image Classification,” IEEE Transactions on 
Image Processing, 2022.

This code is modified based on https://github.com/KGPML/Hyperspectral and https://github.com/xiangyongcao/CNN_HSIC_MRF
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib import slim
import tensorflow as tf
import numpy as np
import os
import scipy.io
import time
import math
from attention import CSpeA
from attention import RSpaA
import copy

# ***************************************select database**********************
# ----------------for Houston 2013 database
#from utils_houston import patch_size, Prepare_data, Height, Width, Band, Database, n_classes, num_band, num_classes
# ----------------for Pavia University database
from utils_PU import patch_size, Prepare_data, Height, Width, Band, Database, n_classes, num_band, num_classes
# ----------------for Salinas database
#from utils_salinas import patch_size, Prepare_data, Height, Width, Band, Database, n_classes, num_band, num_classes

print('===============================================================================')
print('                                   RIAN                                        ')
print('===============================================================================')

start_time = time.time()
Database_path = 'result_RIAN/' + Database 
RESULT_PATH = os.path.join(os.getcwd(),Database_path)
DATA_PATH = os.path.join(os.getcwd(),"Data")

# Import HSI data (Training and Testing samples)
Training_data, Test_data = Prepare_data()
n_input = Band * patch_size * patch_size 
num_train = len(Training_data['train_patch'])
num_test = len(Test_data['test_patch'])

Training_data['train_patch'] = np.transpose(Training_data['train_patch'],(0,2,3,1))  # N*C*H*W -> N*H*W*C
Test_data['test_patch'] = np.transpose(Test_data['test_patch'],(0,2,3,1)) # N*C*H*W -> N*H*W*C


############################# rotate testing samples (testing samples are HSI patches)
# rotate samples from the testing set for evaluating the performance of RIAN under different rotation degrees 
flag_augment_test_rotation = True
# rotate 90
Test_Patch_90 = copy.deepcopy(Test_data['test_patch'])
if flag_augment_test_rotation:
    for k in range(len(Test_data['test_patch'])):
        Test_Patch_90[k] = np.rot90(Test_Patch_90[k], k=1)
# rotate 180
Test_Patch_180 = copy.deepcopy(Test_data['test_patch'])
if flag_augment_test_rotation:
    for k in range(len(Test_data['test_patch'])):
        Test_Patch_180[k] = np.rot90(Test_Patch_180[k], k=2)      
# rotate 270
Test_Patch_270 = copy.deepcopy(Test_data['test_patch'])
if flag_augment_test_rotation:
    for k in range(len(Test_data['test_patch'])):
        Test_Patch_270[k] = np.rot90(Test_Patch_270[k], k=3)  


# Parameters Setting for RIAN
epoch = 200
learning_rate_base = 0.01
learning_rate_decay = 0.5
num_epoch_deacy = 20
batch_size = 100
training_iters = np.ceil(num_train*1.0/batch_size*epoch).astype(int)
print(training_iters)

# tf Graph input
x = tf.placeholder("float", [None, patch_size, patch_size, Band])
y = tf.placeholder("float", [None, n_classes])
is_training = tf.placeholder(tf.bool)

# Define the network framework
def conv_net(x):
    batch_norm_params = {
      'decay': 0.95,
      'epsilon': 1e-5,
      'scale': True,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'is_training': is_training,
      'fused': None,
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params=batch_norm_params):
        #tf.set_random_seed(10)
        Kernel_num = 64
        drop_rate = 0.6
        x_spe = CSpeA(x)
        Encoder1 = slim.conv2d(x_spe, Kernel_num, 1, padding='SAME', scope='Encoder1')

        # block 1
        Encoder2 = slim.conv2d(Encoder1, Kernel_num, 1, padding='SAME', scope='Encoder2')
        Encoder3 = slim.conv2d(Encoder2, Kernel_num, 1, padding='SAME', scope='Encoder3')
        Encoder3_spa = RSpaA(Encoder3, Kernel_num, drop_rate)
        Encoder4 = slim.conv2d(Encoder3_spa, Kernel_num, 1, padding='SAME', scope='Encoder4', activation_fn=None)
        Encoder4_add = tf.nn.relu(tf.add(Encoder1, Encoder4))

        # block 2
        Encoder5 = slim.conv2d(Encoder4_add, Kernel_num, 1, padding='SAME', scope='Encoder5')
        Encoder6 = slim.conv2d(Encoder5, Kernel_num, 1, padding='SAME', scope='Encoder6')
        Encoder6_spa = RSpaA(Encoder6, Kernel_num, drop_rate)
        Encoder7 = slim.conv2d(Encoder6_spa, Kernel_num, 1, padding='SAME', scope='Encoder7', activation_fn=None)
        Encoder7_add = tf.nn.relu(tf.add(Encoder4_add, Encoder7))


        GAP = slim.avg_pool2d(Encoder7_add, patch_size, stride = patch_size, padding='VALID')
        print(GAP.shape)
        GAP_flat = slim.flatten(GAP)
        logits = slim.fully_connected(GAP_flat, num_classes,  activation_fn=None)

    return logits

# Construct model
pred = conv_net(x)
softmax_output= tf.nn.softmax(pred)

# Define loss and optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(cross_entropy)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
                    learning_rate_base,
                    global_step,
                    #np.ceil(num_train*1.0/batch_size*num_epoch_deacy).astype(int),
                    np.ceil((int(math.ceil(num_train / (batch_size*1.0))))*num_epoch_deacy+1).astype(int),
                    learning_rate_decay,
                    staircase=True)
#learning_rate = learning_rate_base
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies([tf.group(*update_ops)]):
    train_op = optimizer.minimize(cost,global_step=global_step)

# Define accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
correct_counter = tf.reduce_sum(tf.cast(correct_prediction, "float"))
#print("Iteraion", '%04d,' % (tf.shape(correct_counter).eval()))
predict_test_label = tf.argmax(pred, 1)

# Initializing the variables
init = tf.global_variables_initializer()

np.random.seed(0)

# Launch the graph
with tf.Session() as sess:
    sess.run(init) 
    # Training cycle
    for epoch_iter in range(epoch):
        # number of batches
        num_batches_per_epoch = int(math.ceil(num_train / (batch_size*1.0)))
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(num_train)
        shuffled_data = Training_data['train_patch'][shuffle_indices,:,:,:]
        shuffled_labels = Training_data['train_labels'][shuffle_indices,:]
        
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size 
            end_index = min((batch_num+1)*batch_size, num_train) 
            # Use the shuffled_data and shuffled_labels for training
            batch_x = shuffled_data[start_index:end_index,:,:,:]
            batch_y = shuffled_labels[start_index:end_index,:]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, batch_cost, train_acc = sess.run([train_op, cost, accuracy], 
                                            feed_dict={x: batch_x,y: batch_y,is_training: True})

        print("Epoch:", '%04d,' % (epoch_iter), \
        "Learning rate=%0.4f," % (learning_rate.eval()), \
        "Batch cost=%.4f," % (batch_cost),\
        "Training Accuracy=%.4f" % (train_acc))


        # Display logs per 10 epochs 
        if ((epoch_iter+1) % 50 == 0):  #and (epoch_iter != 0) 
            # print(x_test.shape)
            # training accuracy
            times_train = int(num_train/batch_size)
            num_each_time_train = int(num_train / times_train)
            res_num_train = num_train - times_train * num_each_time_train
            Num_Each_File_train = num_each_time_train * np.ones((1,times_train),dtype=int)
            Num_Each_File_train = Num_Each_File_train[0]
            Num_Each_File_train[times_train-1] = Num_Each_File_train[times_train-1] + res_num_train
            counter_train = 0
            start_train = 0
            for i in range(times_train):
                feed_x_train = Training_data['train_patch'][start_train:start_train+Num_Each_File_train[i],:,:,:]
                feed_y_train = Training_data['train_labels'][start_train:start_train+Num_Each_File_train[i],:]
                temp_train = sess.run(correct_counter, feed_dict={x: feed_x_train,y: feed_y_train,is_training: False})
                #print('Temp = %.4f' % temp_train)
                counter_train += temp_train
                #print('counter = %.4f' % counter_train)
                start_train += Num_Each_File_train[i]
            accuracy_train = float(counter_train)/float(num_train) 
            print('Training Data Eval: Training Accuracy = %.4f' % accuracy_train)

            # testing accuracy
            times = int(num_test/batch_size)
            num_each_time = int(num_test / times)
            res_num = num_test - times * num_each_time
            Num_Each_File = num_each_time * np.ones((1,times),dtype=int)
            Num_Each_File = Num_Each_File[0]
            Num_Each_File[times-1] = Num_Each_File[times-1] + res_num
            counter = 0
            start = 0
            #temp = 0
            prob_map = np.zeros((1,n_classes))
            for i in range(times):
                feed_x = Test_data['test_patch'][start:start+Num_Each_File[i],:,:,:]
                feed_y = Test_data['test_labels'][start:start+Num_Each_File[i],:]
                temp,temp_map = sess.run([correct_counter,softmax_output], feed_dict={x: feed_x,y: feed_y,is_training: False})
                #print('Test Data Eval: Test Accuracy = %.4f' % temp)    
                #temp_map = sess.run(softmax_output,feed_dict={x: feed_x})
                prob_map = np.concatenate((prob_map,temp_map),axis=0)
                counter = counter + temp
                start += Num_Each_File[i]
            accuracy_test = float(counter)/float(num_test) 
            print('Test Data Eval: Test Accuracy = %.4f' % accuracy_test)
            prob_map = np.delete(prob_map,(0),axis=0)
            #predict_softmax = sess.run(softmax_output,feed_dict={x: x_test})
            print('The shape of prob_map is (%d,%d)' %(prob_map.shape[0],prob_map.shape[1]))   
            prob = {}
            prob['prob_map'] = prob_map
            file_name = 'prob_map_'+str(patch_size)+'_'+str(epoch_iter)+'_'+str(accuracy_test)+'.mat'
            #scipy.io.savemat(os.path.join(RESULT_PATH, file_name),prob)   



        if epoch_iter == epoch-1:
            print("Final epoch", '%04d,' % (epoch_iter))
            #print(x_test.shape)
            # training accuracy
            times_train = int(num_train/batch_size)
            num_each_time_train = int(num_train / times_train)
            res_num_train = num_train - times_train * num_each_time_train
            Num_Each_File_train = num_each_time_train * np.ones((1,times_train),dtype=int)
            Num_Each_File_train = Num_Each_File_train[0]
            Num_Each_File_train[times_train-1] = Num_Each_File_train[times_train-1] + res_num_train
            counter_train = 0
            start_train = 0
            for i in range(times_train):
                feed_x_train = Training_data['train_patch'][start_train:start_train+Num_Each_File_train[i],:,:,:]
                feed_y_train = Training_data['train_labels'][start_train:start_train+Num_Each_File_train[i],:]
                temp_train = sess.run(correct_counter, feed_dict={x: feed_x_train,y: feed_y_train,is_training: False})
                #print('Temp = %.4f' % temp_train)
                counter_train += temp_train
                #print('counter = %.4f' % counter_train)
                start_train += Num_Each_File_train[i]
            accuracy_train = float(counter_train)/float(num_train) 
            print('Training Data Eval: Training Accuracy = %.4f' % accuracy_train)

            # testing accuracy
            times = int(num_test/batch_size)
            num_each_time = int(num_test / times)
            res_num = num_test - times * num_each_time
            Num_Each_File = num_each_time * np.ones((1,times),dtype=int)
            Num_Each_File = Num_Each_File[0]
            Num_Each_File[times-1] = Num_Each_File[times-1] + res_num
            counter = 0
            start = 0
            #temp = 0
            prob_map = np.zeros((1,n_classes))
            for i in range(times):
                feed_x = Test_data['test_patch'][start:start+Num_Each_File[i],:,:,:]
                feed_y = Test_data['test_labels'][start:start+Num_Each_File[i],:]
                temp,temp_map = sess.run([correct_counter,softmax_output], feed_dict={x: feed_x,y: feed_y,is_training: False})
                #print('Test Data Eval: Test Accuracy = %.4f' % temp)    
                #temp_map = sess.run(softmax_output,feed_dict={x: feed_x})
                prob_map = np.concatenate((prob_map,temp_map),axis=0)
                counter = counter + temp
                start += Num_Each_File[i]
            accuracy_test = float(counter)/float(num_test) 
            print('Test Data Eval: Test Accuracy = %.4f' % accuracy_test)
            prob_map = np.delete(prob_map,(0),axis=0)
            #predict_softmax = sess.run(softmax_output,feed_dict={x: x_test})
            print('The shape of prob_map is (%d,%d)' %(prob_map.shape[0],prob_map.shape[1]))   
            prob = {}
            prob['prob_map'] = prob_map
            file_name = 'prob_map_'+str(patch_size)+'_'+str(epoch_iter)+'.mat'
            scipy.io.savemat(os.path.join(RESULT_PATH, file_name),prob)    

            # testing accuracy rotate 90
            times = int(num_test/batch_size)
            num_each_time = int(num_test / times)
            res_num = num_test - times * num_each_time
            Num_Each_File = num_each_time * np.ones((1,times),dtype=int)
            Num_Each_File = Num_Each_File[0]
            Num_Each_File[times-1] = Num_Each_File[times-1] + res_num
            counter = 0
            start = 0
            #temp = 0
            prob_map = np.zeros((1,n_classes))
            for i in range(times):
                feed_x = Test_Patch_90[start:start+Num_Each_File[i],:,:,:]
                feed_y = Test_data['test_labels'][start:start+Num_Each_File[i],:]
                temp,temp_map = sess.run([correct_counter,softmax_output], feed_dict={x: feed_x,y: feed_y,is_training: False})
                #print('Test Data Eval: Test Accuracy = %.4f' % temp)    
                #temp_map = sess.run(softmax_output,feed_dict={x: feed_x})
                prob_map = np.concatenate((prob_map,temp_map),axis=0)
                counter = counter + temp
                start += Num_Each_File[i]
            accuracy_test = float(counter)/float(num_test) 
            print('Test Data Eval rotate 90: Test Accuracy = %.4f' % accuracy_test)
            prob_map = np.delete(prob_map,(0),axis=0)
            #predict_softmax = sess.run(softmax_output,feed_dict={x: x_test})
            print('The shape of prob_map is (%d,%d)' %(prob_map.shape[0],prob_map.shape[1]))   
            prob = {}
            prob['prob_map'] = prob_map
            file_name = 'prob_map_90_'+str(patch_size)+'_'+str(epoch_iter)+'.mat'
            scipy.io.savemat(os.path.join(RESULT_PATH, file_name),prob)   

            # testing accuracy rotate 180
            times = int(num_test/batch_size)
            num_each_time = int(num_test / times)
            res_num = num_test - times * num_each_time
            Num_Each_File = num_each_time * np.ones((1,times),dtype=int)
            Num_Each_File = Num_Each_File[0]
            Num_Each_File[times-1] = Num_Each_File[times-1] + res_num
            counter = 0
            start = 0
            #temp = 0
            prob_map = np.zeros((1,n_classes))
            for i in range(times):
                feed_x = Test_Patch_180[start:start+Num_Each_File[i],:,:,:]
                feed_y = Test_data['test_labels'][start:start+Num_Each_File[i],:]
                temp,temp_map = sess.run([correct_counter,softmax_output], feed_dict={x: feed_x,y: feed_y,is_training: False})
                #print('Test Data Eval: Test Accuracy = %.4f' % temp)    
                #temp_map = sess.run(softmax_output,feed_dict={x: feed_x})
                prob_map = np.concatenate((prob_map,temp_map),axis=0)
                counter = counter + temp
                start += Num_Each_File[i]
            accuracy_test = float(counter)/float(num_test) 
            print('Test Data Eval rotate 180: Test Accuracy = %.4f' % accuracy_test)
            prob_map = np.delete(prob_map,(0),axis=0)
            #predict_softmax = sess.run(softmax_output,feed_dict={x: x_test})
            print('The shape of prob_map is (%d,%d)' %(prob_map.shape[0],prob_map.shape[1]))   
            prob = {}
            prob['prob_map'] = prob_map
            file_name = 'prob_map_180_'+str(patch_size)+'_'+str(epoch_iter)+'.mat'
            scipy.io.savemat(os.path.join(RESULT_PATH, file_name),prob) 

            # testing accuracy rotate 270
            times = int(num_test/batch_size)
            num_each_time = int(num_test / times)
            res_num = num_test - times * num_each_time
            Num_Each_File = num_each_time * np.ones((1,times),dtype=int)
            Num_Each_File = Num_Each_File[0]
            Num_Each_File[times-1] = Num_Each_File[times-1] + res_num
            counter = 0
            start = 0
            #temp = 0
            prob_map = np.zeros((1,n_classes))
            for i in range(times):
                feed_x = Test_Patch_270[start:start+Num_Each_File[i],:,:,:]
                feed_y = Test_data['test_labels'][start:start+Num_Each_File[i],:]
                temp,temp_map = sess.run([correct_counter,softmax_output], feed_dict={x: feed_x,y: feed_y,is_training: False})
                #print('Test Data Eval: Test Accuracy = %.4f' % temp)    
                #temp_map = sess.run(softmax_output,feed_dict={x: feed_x})
                prob_map = np.concatenate((prob_map,temp_map),axis=0)
                counter = counter + temp
                start += Num_Each_File[i]
            accuracy_test = float(counter)/float(num_test) 
            print('Test Data Eval rotate 270: Test Accuracy = %.4f' % accuracy_test)
            prob_map = np.delete(prob_map,(0),axis=0)
            #predict_softmax = sess.run(softmax_output,feed_dict={x: x_test})
            print('The shape of prob_map is (%d,%d)' %(prob_map.shape[0],prob_map.shape[1]))   
            prob = {}
            prob['prob_map'] = prob_map
            file_name = 'prob_map_270'+str(patch_size)+'_'+str(epoch_iter)+'.mat'
            scipy.io.savemat(os.path.join(RESULT_PATH, file_name),prob) 


    print("Optimization Finished!")
    end_time = time.time()
    print('The elapsed time is %.2f' % (end_time-start_time))    
