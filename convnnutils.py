#!/usr/bin/env python

import time
import math
import numpy
import tensorflow as tf
import PlyReader
import utils


def activation_summary(x):
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

def xavier_conv_init(shape):
    return tf.random_normal(shape, stddev=math.sqrt(1.0/(shape[0]*shape[1]* shape[2])))


def xavier_conv3_init(shape):
    return tf.random_normal(shape, stddev=math.sqrt(1.0/(shape[0]*shape[1]* shape[2]*shape[3])))


def xavier_fc_init(shape):
    return tf.random_normal(shape, stddev=math.sqrt(1.0/shape[0]))


def conv2d_layer(name, input_data, shape, wd=0.1):
    with tf.variable_scope(name):
        # Variables created here will be named "name/weights", "name/biases".
        weights = tf.Variable(xavier_conv_init(shape), name='weights')
        if wd>0:
            tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(weights), wd, name='decay'))
        biases = tf.Variable(tf.zeros([shape[3]]), name="biases")
        conv = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')    
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        layer = tf.nn.relu(bias, name=name)
    #activation_summary(layer)
    return layer


def weight_variable(shape, name='weight'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    #initial = xavier_fc_init(shape)
    #return tf.get_variable(name=name , shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    return tf.Variable(initial, name=name)

def conv3d_layer(name, input_data, shape, wd=0.1, strides=[1, 1, 1, 1, 1]):
    with tf.variable_scope(name):
        # Variables created here will be named "name/weights", "name/biases".
        #weights = tf.Variable(xavier_conv3_init(shape), name='weights')
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        #weights = weight_variable(shape)
        #if wd>0:
        #    tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(weights), wd, name='decay'))
        biases = tf.Variable(tf.zeros([shape[4]]), name="biases")
        #conv = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
        
        conv = tf.nn.conv3d(input_data, weights, strides=strides, padding='SAME')    
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        layer = tf.nn.relu(bias, name=name)
    #activation_summary(layer)
    return layer


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')


 
def fc_layer(name, input_data, shape, wd=0.1):
    with tf.variable_scope(name):
        weights = tf.Variable(xavier_fc_init(shape), name='weights')
        if wd>0:
            tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(weights), wd, name='decay'))
        
        biases = tf.Variable(tf.zeros([shape[1]]), name="biases")
        input_flat = tf.reshape(input_data, [-1,shape[0]])
        layer = tf.nn.relu(tf.nn.xw_plus_b(input_flat, weights, biases, name=name))
    #activation_summary(layer)
    return layer





def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def build_graph(data, keep_prob, num_classes, d2=False):
    if d2:
        return build_graph_2d(data, keep_prob, num_classes)
    else:
        return build_graph_3d_5_3_nopool(data, keep_prob, num_classes)


def build_graph_3d_7_5_5_5(data, keep_prob, num_classes, train=True):
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[7, 7, 7, NUM_CHANNELS, 64], strides=[1, 2, 2, 2, 1])
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    #pool1 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')
    
    conv1 = conv3d_layer("conv1", conv0, [5, 5, 5, 64, 128])
    
    pool2 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')
    
    conv2 = conv3d_layer("conv2", pool2, [5, 5, 5, 128, 256])
    
    conv3 = conv3d_layer("conv3", conv2, [5, 5, 5, 256, 512])
    #kk = conv1.get_shape().as_list()[1]
    #kk = kk / 4 
    #pool44 = tf.nn.max_pool3d(conv1, ksize=[1, kk, kk, kk, 1], strides=[1, kk, kk, kk, 1], padding='SAME', name='pool1')
    #pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = conv3.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv3-filters
    
    #fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    W_fc0 = weight_variable([fc0_inputdim, 512])
    b_fc0 = bias_variable([512])
    h_pool2_flat = tf.reshape(conv3, [-1, fc0_inputdim])
    h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)

    
    
    W_fc1 = weight_variable([512, 2048])
    b_fc1 = bias_variable([2048])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc2 = weight_variable([2048, num_classes])
    b_fc2 = bias_variable([num_classes])
    
    regularizers = (tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    if train:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers 
    else:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers, conv0, conv1, h_fc0, h_fc1


def build_graph_3d_5_5_5(data, keep_prob, num_classes, train=True):
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[5, 5, 5, NUM_CHANNELS, 64], strides=[1, 2, 2, 2, 1])
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    #pool1 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')
    
    conv1 = conv3d_layer("conv1", conv0, [5, 5, 5, 64, 128])
    
    pool2 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')
    
    conv2 = conv3d_layer("conv2", pool2, [5, 5, 5, 128, 128])
    #kk = conv1.get_shape().as_list()[1]
    #kk = kk / 4 
    #pool44 = tf.nn.max_pool3d(conv1, ksize=[1, kk, kk, kk, 1], strides=[1, kk, kk, kk, 1], padding='SAME', name='pool1')
    #pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = conv2.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv3-filters
    
    #fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    W_fc0 = weight_variable([fc0_inputdim, 512])
    b_fc0 = bias_variable([512])
    h_pool2_flat = tf.reshape(conv2, [-1, fc0_inputdim])
    h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)

    
    
    W_fc1 = weight_variable([512, 2048])
    b_fc1 = bias_variable([2048])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc2 = weight_variable([2048, num_classes])
    b_fc2 = bias_variable([num_classes])
    
    regularizers = (tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    if train:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers 
    else:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers, conv0, conv1, h_fc0, h_fc1


def build_graph_3d_7_5_3_nopool(data, keep_prob, num_classes, train=True):
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[7, 7, 7, NUM_CHANNELS, 64], strides=[1, 2, 2, 2, 1])
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    #pool1 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')
    
    conv1 = conv3d_layer("conv1", conv0, [5, 5, 5, 64, 128])
    
    pool2 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')
    
    conv2 = conv3d_layer("conv2", pool2, [3, 3, 3, 128, 128])
    #kk = conv1.get_shape().as_list()[1]
    #kk = kk / 4 
    #pool44 = tf.nn.max_pool3d(conv1, ksize=[1, kk, kk, kk, 1], strides=[1, kk, kk, kk, 1], padding='SAME', name='pool1')
    #pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = conv2.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv3-filters
    
    #fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    W_fc0 = weight_variable([fc0_inputdim, 512])
    b_fc0 = bias_variable([512])
    h_pool2_flat = tf.reshape(conv2, [-1, fc0_inputdim])
    h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)

    
    
    W_fc1 = weight_variable([512, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])
    
    regularizers = (tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    if train:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers 
    else:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers, conv0, conv1, h_fc0, h_fc1


def build_graph_3d_5_3_nopool(data, keep_prob, num_classes, train=True):
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[5, 5, 5, NUM_CHANNELS, 32], strides=[1, 2, 2, 2, 1])
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    kk = conv0.get_shape().as_list()[1]
    kk = kk / 4 
    pool44 = tf.nn.max_pool3d(conv0, ksize=[1, kk, kk, kk, 1], strides=[1, kk, kk, kk, 1], padding='SAME', name='pool1')

    conv1 = conv3d_layer("conv1", conv0, [3, 3, 3, 32, 32])
    
    #pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = conv1.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv3-filters
    
    #fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    W_fc0 = weight_variable([fc0_inputdim, 512])
    b_fc0 = bias_variable([512])
    h_pool2_flat = tf.reshape(conv1, [-1, fc0_inputdim])
    h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)

    
    
    W_fc1 = weight_variable([512, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])
    
    regularizers = (tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    if train:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers 
    else:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers, conv0, pool44, h_fc0, h_fc1

def build_graph_3d_5_3(data, keep_prob, num_classes, train=True):
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[5, 5, 5, NUM_CHANNELS, 32])
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    
    conv1 = conv3d_layer("conv1", conv0, [3, 3, 3, 32, 32])
    kk = conv1.get_shape().as_list()[1]
    kk = kk / 4 
    pool44 = tf.nn.max_pool3d(conv1, ksize=[1, kk, kk, kk, 1], strides=[1, kk, kk, kk, 1], padding='SAME', name='pool1')
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = pool1.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv3-filters
    
    #fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    W_fc0 = weight_variable([fc0_inputdim, 512])
    b_fc0 = bias_variable([512])
    h_pool2_flat = tf.reshape(pool1, [-1, fc0_inputdim])
    h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)

    
    
    W_fc1 = weight_variable([512, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])
    
    regularizers = (tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    if train:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers 
    else:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers, pool44, pool1, h_fc0, h_fc1


def build_graph_3d_5_5_3(data, keep_prob, num_classes, train=True):
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[5, 5, 5, NUM_CHANNELS, 32])
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    
    conv1 = conv3d_layer("conv1",conv0,[5, 5, 5, 32, 32])

    conv2 = conv3d_layer("conv2", conv1, [3, 3, 3, 32, 32])
    kk = conv2.get_shape().as_list()[1]
    kk = kk / 4 
    pool44 = tf.nn.max_pool3d(conv2, ksize=[1, kk, kk, kk, 1], strides=[1, kk, kk, kk, 1], padding='SAME', name='pool1')
    pool1 = tf.nn.max_pool3d(conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = pool1.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv3-filters
    
    #fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    W_fc0 = weight_variable([fc0_inputdim, 512])
    b_fc0 = bias_variable([512])
    h_pool2_flat = tf.reshape(pool1, [-1, fc0_inputdim])
    h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)

    
    
    W_fc1 = weight_variable([512, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])
    
    regularizers = (tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    if train:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers 
    else:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers, pool44, pool1, h_fc0, h_fc1

def build_graph_3d_5_5_3_mine(data, keep_prob, num_classes, train=True):
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[5, 5, 5, NUM_CHANNELS, 32])
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    
    conv1 = conv3d_layer("conv1",conv0,[5, 5, 5, 32, 32])
    conv2 = conv3d_layer("conv2", conv1, [3, 3, 3, 32, 32])
    kk = conv2.get_shape().as_list()[1]
    kk = kk / 4 
    pool44 = tf.nn.max_pool3d(conv2, ksize=[1, kk, kk, kk, 1], strides=[1, kk, kk, kk, 1], padding='SAME', name='pool1')
    pool1 = tf.nn.max_pool3d(conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = pool1.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv3-filters
    
    #fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    W_fc0 = weight_variable([fc0_inputdim, 512])
    #W_fc0 = tf.Variable(xavier_fc_init([fc0_inputdim, 512]), name='weight')
    b_fc0 = bias_variable([512])
    h_pool2_flat = tf.reshape(pool1, [-1, fc0_inputdim])
    h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)

    
    W_fc1 = weight_variable([512, 1024])
    #W_fc1 = tf.Variable(xavier_fc_init([512, 1024]), name='weight')
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc2 = weight_variable([1024, num_classes])
    #W_fc2 = tf.Variable(xavier_fc_init([1024, num_classes]), name='weight')
    b_fc2 = bias_variable([num_classes])
    
    regularizers = (tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    if train:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers 
    else:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers, pool44, pool1, h_fc0, h_fc1


def build_graph_3d_7_3(data, keep_prob, num_classes, train=True):
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[7, 7, 7, NUM_CHANNELS, 32])
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    
    #conv1 = conv3d_layer("conv1",conv0,[5, 5, 5, 32, 32])

    conv2 = conv3d_layer("conv2", conv0, [3, 3, 3, 32, 32])
    kk = conv2.get_shape().as_list()[1]
    kk = kk / 4 
    pool44 = tf.nn.max_pool3d(conv2, ksize=[1, kk, kk, kk, 1], strides=[1, kk, kk, kk, 1], padding='SAME', name='pool1')
    pool1 = tf.nn.max_pool3d(conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = pool1.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv3-filters
    
    #fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    W_fc0 = weight_variable([fc0_inputdim, 512])
    b_fc0 = bias_variable([512])
    h_pool2_flat = tf.reshape(pool1, [-1, fc0_inputdim])
    h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)

    
    
    W_fc1 = weight_variable([512, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc2 = weight_variable([1024, num_classes])
    b_fc2 = bias_variable([num_classes])
    
    regularizers = (tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    if train:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers 
    else:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers, pool44, pool1, h_fc0, h_fc1


def build_graph_3d_7_3_mine(data, keep_prob, num_classes, train=True):
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[7, 7, 7, NUM_CHANNELS, 32])
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    
    #conv1 = conv3d_layer("conv1",conv0,[5, 5, 5, 32, 32])
    conv2 = conv3d_layer("conv2", conv0, [3, 3, 3, 32, 32])
    kk = conv2.get_shape().as_list()[1]
    kk = kk / 4 
    pool44 = tf.nn.max_pool3d(conv2, ksize=[1, kk, kk, kk, 1], strides=[1, kk, kk, kk, 1], padding='SAME', name='pool1')
    pool1 = tf.nn.max_pool3d(conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = pool1.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv3-filters
    
    #fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    W_fc0 = weight_variable([fc0_inputdim, 512])
    #W_fc0 = tf.Variable(xavier_fc_init([fc0_inputdim, 512]), name='weight')
    b_fc0 = bias_variable([512])
    h_pool2_flat = tf.reshape(pool1, [-1, fc0_inputdim])
    h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)

    
    W_fc1 = weight_variable([512, 1024])
    #W_fc1 = tf.Variable(xavier_fc_init([512, 1024]), name='weight')
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)
    if train:
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc2 = weight_variable([1024, num_classes])
    #W_fc2 = tf.Variable(xavier_fc_init([1024, num_classes]), name='weight')
    b_fc2 = bias_variable([num_classes])
    
    regularizers = (tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    if train:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers 
    else:
        return tf.matmul(h_fc1, W_fc2) + b_fc2, regularizers, pool44, pool1, h_fc0, h_fc1

def build_graph_3_3_512(data, keep_prob, num_classes, train=True):
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[3, 3, 3, NUM_CHANNELS, 32])
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    
    conv1 = conv3d_layer("conv1", conv0, [3, 3, 3, 32, 32])
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = pool1.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv3-filters
    
    #fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    W_fc0 = weight_variable([fc0_inputdim, 512],  name='fc0/weights')
    b_fc0 = bias_variable([512], name='fc0/bias')
    h_pool2_flat = tf.reshape(pool1, [-1, fc0_inputdim])
    h_fc0 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc0) + b_fc0)
    if train:
        h_fc0 = tf.nn.dropout(h_fc0, keep_prob)
    
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc1 = weight_variable([512, num_classes], name='fc1/weights')
    b_fc1 = bias_variable([num_classes], name='fc0/bias')
    
    regularizers = (tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0) +
                    tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1))
    if train:
        return tf.matmul(h_fc0, W_fc1) + b_fc1, regularizers
    else:
        return tf.matmul(h_fc0, W_fc1) + b_fc1, regularizers, conv1, pool1, h_fc0


def build_graph_2d(data, keep_prob, num_classes):
    data_shape = data.get_shape().as_list();
    NUM_CHANNELS = data_shape[3]
    conv0 = conv2d_layer("conv0",data,[3, 3, NUM_CHANNELS, 32])
    pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')
    conv1 = conv2d_layer("conv1",pool0,[3, 3, 32, 32])
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[3, 3, 32, 32])
    
    shape = pool1.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3];   # Resolve input dim into fc0 from conv2-filters
    
    fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 1024])   
    
    fc0_drop = tf.nn.dropout(fc0, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc1 = weight_variable([1024, num_classes])
    b_fc1 = bias_variable([num_classes])
    return tf.matmul(fc0_drop, W_fc1) + b_fc1
    #y_conv=tf.nn.softmax(tf.matmul(fc0_drop, W_fc1) + b_fc1)
     
    #return y_conv #tf.reshape(y_conv, [data_shape[0],data_shape[1],data_shape[2]])
