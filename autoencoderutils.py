import numpy as np
import tensorflow as tf
import math
import cv2
#from convnnutils import *


def xavier_conv3_init(shape):
    return tf.random_normal(shape, stddev=math.sqrt(1.0/(shape[0]*shape[1]* shape[2]*shape[3])))


def xavier_fc_init(shape):
    return tf.random_normal(shape, stddev=math.sqrt(1.0/shape[0]))

def conv3d_layer(name, input_data, shape, wd=0.1, strides=[1, 1, 1, 1, 1], padding='SAME'):
    with tf.variable_scope(name):
        weights = tf.Variable(xavier_conv3_init(shape), name='weights')
        if wd>0:
            tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(weights), wd, name='decay'))
        biases = tf.Variable(tf.zeros([shape[4]]), name="biases")
        conv = tf.nn.conv3d(input_data, weights, strides=strides, padding=padding)    
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        layer = tf.nn.relu(bias, name=name)
    #activation_summary(layer)
    return layer, weights

def conv3d_layer_transpose(name, input_data, weights, output_shape, wd=0.1, strides=[1, 1, 1, 1, 1], padding='SAME'):
    with tf.variable_scope(name):
        biases = tf.Variable(tf.zeros([weights.get_shape().as_list()[3]]), name="biases")  
        conv = tf.nn.conv3d_transpose(input_data, weights, output_shape=output_shape, strides=strides, padding=padding)
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list(), name="bias_reshape")
        layer = tf.nn.relu(bias, name=name)
    #activation_summary(layer)
    return layer

def fc_layer(name, input_data, shape, wd=0.1):
    with tf.variable_scope(name):
        weights = tf.Variable(xavier_fc_init(shape), name='weights')
        #weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        if wd>0:
            tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(weights), wd, name='decay'))
        
        biases = tf.Variable(tf.zeros([shape[1]]), name="biases")
        input_flat = tf.reshape(input_data, [-1,shape[0]])
        layer = tf.nn.relu(tf.nn.xw_plus_b(input_flat, weights, biases, name=name))
    #activation_summary(layer)
    return layer

# Function that constructs graph from first conv-layer to last conv-layer


def autoencoder_5_5_draft(net_in, LATENT_DIMS):
    data_shape = net_in.get_shape().as_list()
    NUM_CHANNELS = data_shape[4]
    strides = [1, 2, 2, 2, 1]
    #strides = [1, 1, 1, 1, 1]
    # Define all encoding layers
    conv0, W0 = conv3d_layer("conv0",net_in,[3, 3, 3, NUM_CHANNELS, 32], strides=strides)
    #pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    conv1, W1 = conv3d_layer("conv1",conv0,[3, 3, 3, 32, 64], strides=strides)
    #pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   

    # Resolve input dim into fc0 from pool1-output
    #LATENT_DIMS = 100
    shape = conv1.get_shape().as_list()
    print "conv1 shape: ", shape
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4]
    fc0 = fc_layer("fc0", conv1, [fc0_inputdim, LATENT_DIMS])

    # Start going back by first reshaping into 4D image data again
    # Then two sets of depooling and convolutions
    fc1 = fc_layer("fc1", fc0, [LATENT_DIMS,fc0_inputdim])
    print "fc1 shape: ", fc1.get_shape().as_list()
    fc1_reshaped = tf.reshape(fc1, conv1.get_shape().as_list())
    #depool0 = tf.image.resize_images(fc1_reshaped, [shape[1]*2, shape[2]*2, shape[3]*2])
    #deconv0 = conv3d_layer("deconv0", fc1_reshaped, [3, 3, 3, shape[4], 32])
    print "fc1_reshaped shape: ", fc1_reshaped.get_shape().as_list()
    print "conv0 shape: ", conv0.get_shape().as_list()
    #deconv0 = conv3d_layer_transpose("deconv0", fc1_reshaped, [3, 3, 3, shape[4], 32], output_shape=tf.pack(conv0.get_shape().as_list()))
    deconv0 = conv3d_layer_transpose("deconv0", fc1_reshaped, W1, output_shape=tf.shape(conv0), strides=strides)
    deconv0_shape = deconv0.get_shape().as_list()
    print "deconv0 shape: ", deconv0_shape
    #depool1 = tf.image.resize_images(deconv0, [deconv0_shape[1]*2, deconv0_shape[2]*2, deconv0_shape[3]*2])
    #deconv1 = conv3d_layer("deconv1", deconv0, [3, 3, 3, deconv0_shape[4], NUM_CHANNELS])
    deconv1 = conv3d_layer_transpose("deconv1", deconv0, W0, output_shape=tf.shape(net_in), strides=strides)
    
    return deconv1


def autoencoder_5_5(net_in, LATENT_DIMS):
    data_shape = net_in.get_shape().as_list()
    NUM_CHANNELS = data_shape[4]
    strides = [1, 2, 2, 2, 1]
    # Define all encoding layers
    c0 = 5
    c1 = 5
    padding="VALID"
    conv0, W0 = conv3d_layer("conv0",net_in,[c0, c0, c0, NUM_CHANNELS, 64], strides=strides, padding=padding)
    conv1, W1 = conv3d_layer("conv1",conv0,[c1, c1, c1, 64, 128], strides=strides, padding=padding)   

    # Resolve input dim into fc0 from pool1-output
    #LATENT_DIMS = 100
    shape = conv1.get_shape().as_list()
    print "conv1 shape: ", shape
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4]
    fc0 = fc_layer("fc0", conv1, [fc0_inputdim, LATENT_DIMS])

    # Start going back by first reshaping into 4D image data again
    # Then two sets of depooling and convolutions
    fc1 = fc_layer("fc1", fc0, [LATENT_DIMS,fc0_inputdim])
    fc1_reshaped = tf.reshape(fc1, conv1.get_shape().as_list())
    
    deconv0 = conv3d_layer_transpose("deconv0", fc1_reshaped, W1, output_shape=tf.shape(conv0), strides=strides, padding=padding)
    deconv1 = conv3d_layer_transpose("deconv1", deconv0, W0, output_shape=tf.shape(net_in), strides=strides, padding=padding)
    
    return deconv1


def autoencoder_3_3(net_in, LATENT_DIMS):
    data_shape = net_in.get_shape().as_list()
    NUM_CHANNELS = data_shape[4]
    strides = [1, 2, 2, 2, 1]
    # Define all encoding layers
    c0 = 3
    c1 = 3
    #padding="VALID"
    conv0, W0_old = conv3d_layer("conv0",net_in,[c0, c0, c0, NUM_CHANNELS, 32], strides=strides)
    conv1, W1_old = conv3d_layer("conv1",conv0,[c1, c1, c1, 32, 64], strides=strides)   

    W0= tf.Variable(xavier_conv3_init(W0_old.get_shape().as_list()), name='weights_deconv')
    W1 = tf.Variable(xavier_conv3_init(W1_old.get_shape().as_list()), name='weights_deconv')
    
    # Resolve input dim into fc0 from pool1-output
    #LATENT_DIMS = 100
    shape = conv1.get_shape().as_list()
    print "conv1 shape: ", shape
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4]
    fc0 = fc_layer("fc0", conv1, [fc0_inputdim, LATENT_DIMS])

    # Start going back by first reshaping into 4D image data again
    # Then two sets of depooling and convolutions
    fc1 = fc_layer("fc1", fc0, [LATENT_DIMS,fc0_inputdim])
    fc1_reshaped = tf.reshape(fc1, conv1.get_shape().as_list())
    
    deconv0 = conv3d_layer_transpose("deconv0", fc1_reshaped, W1, output_shape=tf.shape(conv0), strides=strides)
    deconv1 = conv3d_layer_transpose("deconv1", deconv0, W0, output_shape=tf.shape(net_in), strides=strides)
    
    return deconv1


def autoencoder_3_3_3(net_in, LATENT_DIMS):
    data_shape = net_in.get_shape().as_list()
    NUM_CHANNELS = data_shape[4]
    strides = [1, 2, 2, 2, 1]
    # Define all encoding layers
    c0 = 3
    c1 = 3
    c2 = 3
    conv0, W0 = conv3d_layer("aeconv0",net_in,[c0, c0, c0, NUM_CHANNELS, 32], strides=strides)
    conv1, W1 = conv3d_layer("aeconv1",conv0,[c1, c1, c1, 32, 64], strides=strides)   
    conv2, W2 = conv3d_layer("aeconv2",conv1,[c2, c2, c2, 64, 64], strides=[1, 1, 1, 1, 1])
    
    #W0 = tf.Variable(xavier_conv3_init(W0_old.get_shape().as_list()), name='weights')
    #W1 = tf.Variable(xavier_conv3_init(W1_old.get_shape().as_list()), name='weights')
    #W2 = tf.Variable(xavier_conv3_init(W2_old.get_shape().as_list()), name='weights')
    
    # Resolve input dim into fc0 from pool1-output
    #LATENT_DIMS = 100
    shape = conv2.get_shape().as_list()
    print "conv2 shape: ", shape
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4]
    fc0 = fc_layer("aefc0", conv2, [fc0_inputdim, LATENT_DIMS])

    # Start going back by first reshaping into 4D image data again
    # Then two sets of depooling and convolutions
    fc1 = fc_layer("aefc1", fc0, [LATENT_DIMS,fc0_inputdim])
    fc1_reshaped = tf.reshape(fc1, conv2.get_shape().as_list())
    
    deconv0 = conv3d_layer_transpose("aedeconv0", fc1_reshaped, W2, output_shape=tf.shape(conv1), strides=[1, 1, 1, 1, 1])
    deconv1 = conv3d_layer_transpose("aedeconv1", deconv0, W1, output_shape=tf.shape(conv0), strides=strides)
    deconv2 = conv3d_layer_transpose("aedeconv2", deconv1, W0, output_shape=tf.shape(net_in), strides=strides)
    
    return deconv2


def autoencoder_3_3_3_3(net_in, LATENT_DIMS):
    data_shape = net_in.get_shape().as_list()
    NUM_CHANNELS = data_shape[4]
    strides = [1, 2, 2, 2, 1]
    # Define all encoding layers
    c0 = 3
    c1 = 3
    c2 = 3
    c3 = 3
    conv0, W0 = conv3d_layer("aeconv0",net_in,[c0, c0, c0, NUM_CHANNELS, 32], strides=strides)
    conv1, W1 = conv3d_layer("aeconv1",conv0,[c1, c1, c1, 32, 64], strides=strides)   
    conv2, W2 = conv3d_layer("aeconv2",conv1,[c2, c2, c2, 64, 64], strides=[1, 1, 1, 1, 1])
    conv3, W3 = conv3d_layer("aeconv2",conv2,[c3, c3, c3, 64, 64], strides=[1, 1, 1, 1, 1])
    
    #W0 = tf.Variable(xavier_conv3_init(W0_old.get_shape().as_list()), name='weights')
    #W1 = tf.Variable(xavier_conv3_init(W1_old.get_shape().as_list()), name='weights')
    #W2 = tf.Variable(xavier_conv3_init(W2_old.get_shape().as_list()), name='weights')
    
    # Resolve input dim into fc0 from pool1-output
    #LATENT_DIMS = 100
    shape = conv3.get_shape().as_list()
    print "conv3 shape: ", shape
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4]
    fc0 = fc_layer("aefc0", conv3, [fc0_inputdim, LATENT_DIMS])

    # Start going back by first reshaping into 4D image data again
    # Then two sets of depooling and convolutions
    fc1 = fc_layer("aefc1", fc0, [LATENT_DIMS,fc0_inputdim])
    fc1_reshaped = tf.reshape(fc1, conv2.get_shape().as_list())
    
    
    deconv0 = conv3d_layer_transpose("aedeconv0", fc1_reshaped, W3, output_shape=tf.shape(conv2), strides=[1, 1, 1, 1, 1])
    deconv1 = conv3d_layer_transpose("aedeconv1", deconv0, W2, output_shape=tf.shape(conv1), strides=[1, 1, 1, 1, 1])
    deconv2 = conv3d_layer_transpose("aedeconv2", deconv1, W1, output_shape=tf.shape(conv0), strides=strides)
    deconv3 = conv3d_layer_transpose("aedeconv3", deconv2, W0, output_shape=tf.shape(net_in), strides=strides)
    
    return deconv3, fc0


def autoencoder_3_3_3_3_3(net_in, LATENT_DIMS):
    data_shape = net_in.get_shape().as_list()
    NUM_CHANNELS = data_shape[4]
    strides = [1, 2, 2, 2, 1]
    # Define all encoding layers
    c0 = 5
    c1 = 3
    c2 = 3
    c3 = 3
    c4 = 3
    conv0, W0 = conv3d_layer("aeconv0",net_in,[c0, c0, c0, NUM_CHANNELS, 32], strides=strides)
    conv1, W1 = conv3d_layer("aeconv1",conv0,[c1, c1, c1, 32, 64], strides=strides)   
    conv2, W2 = conv3d_layer("aeconv2",conv1,[c2, c2, c2, 64, 64], strides=[1, 1, 1, 1, 1])
    conv3, W3 = conv3d_layer("aeconv3",conv2,[c3, c3, c3, 64, 64], strides=[1, 1, 1, 1, 1])
    conv4, W4 = conv3d_layer("aeconv4",conv3,[c3, c3, c3, 64, 64], strides=[1, 1, 1, 1, 1])
    
    #W0 = tf.Variable(xavier_conv3_init(W0_old.get_shape().as_list()), name='weights')
    #W1 = tf.Variable(xavier_conv3_init(W1_old.get_shape().as_list()), name='weights')
    #W2 = tf.Variable(xavier_conv3_init(W2_old.get_shape().as_list()), name='weights')
    
    # Resolve input dim into fc0 from pool1-output
    #LATENT_DIMS = 100
    shape = conv4.get_shape().as_list()
    print "conv4 shape: ", shape
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4]
    fc0 = fc_layer("aefc0", conv4, [fc0_inputdim, LATENT_DIMS])

    # Start going back by first reshaping into 4D image data again
    # Then two sets of depooling and convolutions
    fc1 = fc_layer("aefc1", fc0, [LATENT_DIMS,fc0_inputdim])
    fc1_reshaped = tf.reshape(fc1, shape)
    
    
    deconv0 = conv3d_layer_transpose("aedeconv0", fc1_reshaped, W4, output_shape=tf.shape(conv3), strides=[1, 1, 1, 1, 1])
    deconv1 = conv3d_layer_transpose("aedeconv1", deconv0, W3, output_shape=tf.shape(conv2), strides=[1, 1, 1, 1, 1])
    deconv2 = conv3d_layer_transpose("aedeconv2", deconv1, W2, output_shape=tf.shape(conv1), strides=[1, 1, 1, 1, 1])
    deconv3 = conv3d_layer_transpose("aedeconv3", deconv2, W1, output_shape=tf.shape(conv0), strides=strides)
    deconv4 = conv3d_layer_transpose("aedeconv4", deconv3, W0, output_shape=tf.shape(net_in), strides=strides)
    
    return deconv4, fc0

