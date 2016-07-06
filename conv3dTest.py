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
from tensorflow.examples.tutorials.mnist import input_data
import cv2
from matplotlib import pyplot as plt

IMAGE_SIZE = 28
NUM_CHANNELS = 1

def fake_data(num_images):
    """Generate a fake dataset that matches the dimensions of MNIST."""
    data = numpy.ndarray(
        shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
        dtype=numpy.float32)
    labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
    for image in xrange(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image] = label
    return data, labels


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2x2(x):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
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
    W_conv1 = weight_variable([5, 5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv3d(data, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 5, 32, 64])
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
   
    regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
      tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    return tf.matmul(h_fc1_drop, W_fc2) + b_fc2, regularizers


def build_graph_2d(data, keep_prob, num_classes):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(data, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    shape = h_pool2.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]  # Resolve input dim into fc0 from conv2-filters
    
    W_fc1 = weight_variable([fc0_inputdim, 512])
    b_fc1 = bias_variable([512])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, fc0_inputdim])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([512, num_classes])
    b_fc2 = bias_variable([num_classes])

    regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
          tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    return tf.matmul(h_fc1_drop, W_fc2) + b_fc2, regularizers

SEED = None
#NUM_LABELS = 2


import os.path

def main():
    
    
    nr_epochs = 500
    
    BATCH_SIZE = 20
    num_rotations = 10
    patch_dim = 28
    relL = 0.07
    
    
    dir1 = os.path.dirname(os.path.realpath(__file__))
        
    fileName = os.path.join(dir1, 'plytest/bun_zipper.ply')
    reader = PlyReader.PlyReader()
    start_time = time.time()
    reader.read_ply(fileName, num_samples=10)
    print 'reading time: ', time.time() - start_time
    pc_diameter = utils.get_pc_diameter(reader.data)
    l = relL*pc_diameter
    reader.set_variables(l=l, patch_dim=patch_dim)
    samples_count = reader.compute_total_samples(num_rotations)
    batches_per_epoch = samples_count/BATCH_SIZE
    print "Batches per epoch:", batches_per_epoch
    
    
    
    num_epochs = 10000
    train_size = 100
    train_data, train_labels = fake_data(train_size)
    
    train_data, train_labels = reader.next_batch(train_size // num_rotations, num_rotations=num_rotations, num_channels=1, d2 = True)
    ii = numpy.random.permutation(train_labels.shape[0])
    #train_data = train_data[ii]
    #train_labels = train_labels[ii]
    NUM_LABELS = reader.num_classes
    print 'NUM_LABELS: ', NUM_LABELS
    offset = 0
    X = train_data[offset:(offset + BATCH_SIZE), ...]
    Y = train_labels[offset:(offset + BATCH_SIZE)]
    
    #X, Y = fake_data(BATCH_SIZE)
    #mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    #X,Y = mnist.train.next_batch(50)
    #X = numpy.reshape(X, [50, 28, 28, 1])
    #Y = numpy.argmax(Y, axis=1)
    #Y = numpy.reshape(Y, [50,])
    
    with tf.Graph().as_default() as graph:
        
        net_x = tf.placeholder("float", X.shape, name="in_x")
        net_y = tf.placeholder(tf.int32, Y.shape, name="in_y")
        
        logits, regularizers = build_graph_2d(net_x, 0.5, NUM_LABELS)
        loss= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                                             logits, net_y))
        #regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
         #         tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        
        loss += 5e-4 * regularizers
        print 'logits shape: ',logits.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
        print 'X shape: ',  net_x.get_shape().as_list()        
        
        global_step = tf.Variable(0, trainable=False)
        
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        #learning_rate = tf.train.exponential_decay(
        #    0.01,                # Base learning rate.
         #   batch * BATCH_SIZE,  # Current index into the dataset.
          #  train_size,          # Decay step.
           # 0.95,                # Decay rate.
            #staircase=True)
        # Use simple momentum for the optimization.
        #optimizer = tf.train.MomentumOptimizer(0.00001, 0.9).minimize(loss, global_step=batch)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss, global_step=batch)
        train_prediction = tf.nn.softmax(logits)
        
        # Create initialization "op" and run it with our session 
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
        sess.run(init)
        
        # Create a saver and a summary op based on the tf-collection
        saver = tf.train.Saver(tf.all_variables())
        
                
        #saver.restore(sess, 'model_bunny_5_10.ckpt')   # Load previously trained weights
                        
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            #offset = 0
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            

            
            feed_dict = {net_x: batch_data,
                   net_y: batch_labels}
            _, l, _ = sess.run(
                                             [optimizer, loss, train_prediction],
                                             feed_dict=feed_dict)
            print 'Step: ', step, '    loss: ', l#, '    lr: ', lr

    print 'done'
if __name__ == "__main__":
    main()


