import numpy as np
import tensorflow as tf
import math
import cv2
from convnnutils import *


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
    conv0, W0 = conv3d_layer("conv0",net_in,[c0, c0, c0, NUM_CHANNELS, 64], strides=strides, padding="VALID")
    conv1, W1 = conv3d_layer("conv1",conv0,[c1, c1, c1, 64, 128], strides=strides, padding="VALID")   

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
    
    deconv0 = conv3d_layer_transpose("deconv0", fc1_reshaped, W1, output_shape=tf.shape(conv0), strides=strides, padding="VALID")
    deconv1 = conv3d_layer_transpose("deconv1", deconv0, W0, output_shape=tf.shape(net_in), strides=strides, padding="VALID")
    
    return deconv1


def autoencoder_3_3(net_in, LATENT_DIMS):
    data_shape = net_in.get_shape().as_list()
    NUM_CHANNELS = data_shape[4]
    strides = [1, 2, 2, 2, 1]
    # Define all encoding layers
    c0 = 3
    c1 = 3
    conv0, W0 = conv3d_layer("conv0",net_in,[c0, c0, c0, NUM_CHANNELS, 32], strides=strides)
    conv1, W1 = conv3d_layer("conv1",conv0,[c1, c1, c1, 32, 64], strides=[1, 1, 1, 1, 1])   

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
    
    deconv0 = conv3d_layer_transpose("deconv0", fc1_reshaped, W1, output_shape=tf.shape(conv0), strides=[1, 1, 1, 1, 1])
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
    conv0, W0 = conv3d_layer("conv0",net_in,[c0, c0, c0, NUM_CHANNELS, 32], strides=strides)
    conv1, W1 = conv3d_layer("conv1",conv0,[c1, c1, c1, 32, 64], strides=strides)   
    conv2, W2 = conv3d_layer("conv2",conv1,[c2, c2, c2, 64, 64], strides=[1, 1, 1, 1, 1])

    # Resolve input dim into fc0 from pool1-output
    #LATENT_DIMS = 100
    shape = conv2.get_shape().as_list()
    print "conv2 shape: ", shape
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4]
    fc0 = fc_layer("fc0", conv2, [fc0_inputdim, LATENT_DIMS])

    # Start going back by first reshaping into 4D image data again
    # Then two sets of depooling and convolutions
    fc1 = fc_layer("fc1", fc0, [LATENT_DIMS,fc0_inputdim])
    fc1_reshaped = tf.reshape(fc1, conv2.get_shape().as_list())
    
    deconv0 = conv3d_layer_transpose("deconv0", fc1_reshaped, W2, output_shape=tf.shape(conv1), strides=[1, 1, 1, 1, 1])
    deconv1 = conv3d_layer_transpose("deconv1", deconv0, W1, output_shape=tf.shape(conv0), strides=strides)
    deconv2 = conv3d_layer_transpose("deconv2", deconv1, W0, output_shape=tf.shape(net_in), strides=strides)
    
    return deconv2
