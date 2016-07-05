#!/usr/bin/env python

import h5py
#import os
#import re
#import sys
#from datetime import datetime
#import os.path
import time
import math
import numpy
#import tensorflow.python.platform
import tensorflow as tf
#import cv2
import PlyReader
import utils
from twisted.test.test_tcp import numRounds
import os



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')
  

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def build_graph(data, keep_prob, num_classes, d2=False):
    if d2:
        return build_graph_2d(data, keep_prob, num_classes)
    else:
        return build_graph_3d(data, keep_prob, num_classes)

def build_graph_3d(data, keep_prob, num_classes):
    W_conv1 = weight_variable([3, 3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv3d(data, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2x2(h_conv1)
    
    W_conv2 = weight_variable([3, 3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2x2(h_conv2)
    
    shape = h_pool2.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4]  # Resolve input dim into fc0 from conv2-filters
    
    W_fc1 = weight_variable([fc0_inputdim, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, fc0_inputdim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    return tf.matmul(h_fc1_drop, W_fc2) + b_fc2


def build_graph_2d(data, keep_prob, num_classes):
    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    shape = h_pool2.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]  # Resolve input dim into fc0 from conv2-filters
    
    W_fc1 = weight_variable([fc0_inputdim, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, fc0_inputdim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    return tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    


import os.path

def main():
    
    d2 = True
    num_channels = 1
    
    INIT_RATE = 0.00001
    LR_DECAY_FACTOR = 0.1
    
    nr_epochs = 500
    
    num_rotations = 5
    patch_dim = 10
    #relSampling = 0.05
    #relRadius = 0.1
    #radius = pc_diameter*relRadius
    #relL = 0.07
    relL = 0.05
    
    BATCH_SIZE = 10
    
    dir1 = os.path.dirname(os.path.realpath(__file__))
	    
    fileName = os.path.join(dir1, 'plytest/bun_zipper.ply')
    reader = PlyReader.PlyReader()
    start_time = time.time()
    reader.read_ply(fileName, num_samples=20)
    print 'reading time: ', time.time() - start_time
    pc_diameter = utils.get_pc_diameter(reader.data)
    l = relL*pc_diameter
    reader.set_variables(l=l, num_rotations=num_rotations, patch_dim=patch_dim)
    samples_count = reader.compute_total_samples(num_rotations)
    batches_per_epoch = samples_count/BATCH_SIZE
    print "Batches per epoch:", batches_per_epoch
    

    
    start_time = time.time()
    X,Y = reader.next_batch(BATCH_SIZE, num_rotations=num_rotations, num_channels=num_channels, d2=d2)

    print 'batch time: ', time.time() - start_time
    
    
    with tf.Graph().as_default() as graph:
        net_x = tf.placeholder("float", X.shape, name="in_x")
        net_y = tf.placeholder(tf.int32, Y.shape, name="in_y")
        
        output = build_graph(net_x, 0.5, reader.num_classes, d2=d2)
        
        print 'output shape: ',output.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
        print 'X shape: ',  net_x.get_shape().as_list()
        #assert (output.get_shape().as_list() == net_y.get_shape().as_list() )
        
        global_step = tf.Variable(0, trainable=False)
        
        
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output, net_y))
        decay_steps = int(batches_per_epoch)
        lr = tf.train.exponential_decay(INIT_RATE, global_step, decay_steps, LR_DECAY_FACTOR,  staircase=True)
        opt = tf.train.GradientDescentOptimizer(0.5)
        train_op = opt.minimize(cross_entropy)
        
        # Create initialization "op" and run it with our session 
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
        sess.run(init)
        
        # Create a saver and a summary op based on the tf-collection
        saver = tf.train.Saver(tf.all_variables())
        
                
        #saver.restore(sess, 'model_bunny_5_10.ckpt')   # Load previously trained weights
                        
        for epoch in range(nr_epochs):
                print "Starting epoch ", epoch
                for batch in range(batches_per_epoch):
                    X, Y= reader.next_batch(BATCH_SIZE, num_rotations=num_rotations, num_channels=num_channels, d2=d2)
                
                    start_time = time.time()
                    _, error = sess.run([train_op, cross_entropy], feed_dict={net_x:X, net_y: Y})
                    duration = time.time() - start_time
                    print "Batch:", batch ,"  Loss: ", error, "   Duration (sec): ", duration

                    if batch % 100 == 0:
                        save_path = saver.save(sess, "model_bunny_5_10_test.ckpt")
                        print "Model saved in file: ",save_path

    print 'done'
if __name__ == "__main__":
    main()


