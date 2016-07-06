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

SEED = None
NUM_LABELS = 2




import os.path

def main():
    

    nr_epochs = 500
    
    BATCH_SIZE = 16
    num_epochs = 10
    train_size = 256
    train_data, train_labels = fake_data(train_size)
    
    offset = (0 * BATCH_SIZE) % (train_size - BATCH_SIZE)
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
        
        conv1_weights = tf.Variable(
                                    tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
        conv1_biases = tf.Variable(tf.zeros([32]))
        conv2_weights = tf.Variable(
          tf.truncated_normal([5, 5, 32, 64],
                              stddev=0.1,
                              seed=SEED))
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
        fc1_weights = tf.Variable(  # fully connected, depth 512.
          tf.truncated_normal(
              [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
              stddev=0.1,
              seed=SEED))
        fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
        fc2_weights = tf.Variable(
          tf.truncated_normal([512, NUM_LABELS],
                              stddev=0.1,
                              seed=SEED))
        fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
      
        def model(data, train=False):
            conv = tf.nn.conv2d(data,
                                conv1_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            # Bias and rectified linear non-linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
            # Max pooling. The kernel size spec {ksize} also follows the layout of
            # the data. Here we have a pooling window of 2, and a stride of 2.
            pool = tf.nn.max_pool(relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')
            conv = tf.nn.conv2d(pool,
                                conv2_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
            pool = tf.nn.max_pool(relu,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')
            # Reshape the feature map cuboid into a 2D matrix to feed it to the
            # fully connected layers.
            pool_shape = pool.get_shape().as_list()
            reshape = tf.reshape(
                pool,
                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
            # Add a 50% dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            if train:
              hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
            return tf.matmul(hidden, fc2_weights) + fc2_biases
        
        logits = model(net_x, True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                                             logits, net_y))
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        
        loss += 5e-4 * regularizers
        print 'logits shape: ',logits.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
        print 'X shape: ',  net_x.get_shape().as_list()        
        
        global_step = tf.Variable(0, trainable=False)
        
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            0.01,                # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size,          # Decay step.
            0.95,                # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               0.9).minimize(loss,
                                                             global_step=batch)
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
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {net_x: batch_data,
                   net_y: batch_labels}
            _, l, lr, _ = sess.run(
                                             [optimizer, loss, learning_rate, train_prediction],
                                             feed_dict=feed_dict)
            print 'Step: ', step, '    loss: ', l, '    lr: ', lr

    print 'done'
if __name__ == "__main__":
    main()


