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

def conv3d_layer(name, input_data, shape, wd=0.1):
    with tf.variable_scope(name):
        # Variables created here will be named "name/weights", "name/biases".
        weights = tf.Variable(xavier_conv3_init(shape), name='weights')
        if wd>0:
            tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(weights), wd, name='decay'))
        biases = tf.Variable(tf.zeros([shape[4]]), name="biases")
        #conv = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
        
        conv = tf.nn.conv3d(input_data, weights, strides=[1, 1, 1, 1, 1], padding='SAME')    
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        layer = tf.nn.relu(bias, name=name)
    #activation_summary(layer)
    return layer

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
    data_shape = data.get_shape().as_list()
    print 'data shape: ', data_shape
    NUM_CHANNELS = data_shape[4]
    conv0 = conv3d_layer("conv0",data,[5, 5, 5, NUM_CHANNELS, 64])
    #pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')
    pool0 = tf.nn.max_pool3d(conv0, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool0')
    
    conv1 = conv3d_layer("conv1", pool0, [5, 5, 5, 64, 64])
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')   
    #conv2 = conv2d_layer("conv2",pool1,[12, 12, 32, 256])
    
    shape = pool1.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3]*shape[4];   # Resolve input dim into fc0 from conv2-filters
    
    fc0 = fc_layer("fc0", pool1, [fc0_inputdim, 128])   
    
    fc0_drop = tf.nn.dropout(fc0, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc1 = weight_variable([128, num_classes])
    b_fc1 = bias_variable([num_classes])

    y_conv=tf.nn.softmax(tf.matmul(fc0_drop, W_fc1) + b_fc1)
                                                                                                                                                                                                                                                
    return y_conv #tf.reshape(y_conv, [data_shape[0],data_shape[1],data_shape[2]])

def build_graph_2d(data, keep_prob, num_classes):
    data_shape = data.get_shape().as_list();
    NUM_CHANNELS = data_shape[3]
    conv0 = conv2d_layer("conv0",data,[5, 5, NUM_CHANNELS, 64])
    pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')
    conv1 = conv2d_layer("conv1",pool0,[5, 5, 64, 128])
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')   
    conv2 = conv2d_layer("conv2",pool1,[5, 5, 128, 64])
    
    shape = conv2.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3];   # Resolve input dim into fc0 from conv2-filters
    
    fc0 = fc_layer("fc0", conv2, [fc0_inputdim, 512])   
    
    fc0_drop = tf.nn.dropout(fc0, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc1 = weight_variable([512, num_classes])
    b_fc1 = bias_variable([num_classes])

    y_conv=tf.nn.softmax(tf.matmul(fc0_drop, W_fc1) + b_fc1)
     
    return y_conv #tf.reshape(y_conv, [data_shape[0],data_shape[1],data_shape[2]])


def build_graph_orig(data, keep_prob):
    data_shape = data.get_shape().as_list();
    NUM_LABELS = 1000
    NUM_CHANNELS = data_shape[3]
    conv0 = conv2d_layer("conv0",data,[5, 5, NUM_CHANNELS, 64])
    pool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool0')
    conv1 = conv2d_layer("conv1",pool0,[5, 5, 64, 128])
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')   
    conv2 = conv2d_layer("conv2",pool1,[5, 5, 128, 64])
    
    shape = conv2.get_shape().as_list()
    fc0_inputdim = shape[1]*shape[2]*shape[3];   # Resolve input dim into fc0 from conv2-filters
    
    fc0 = fc_layer("fc0", conv2, [fc0_inputdim, 512])   
    
    fc0_drop = tf.nn.dropout(fc0, keep_prob)
    
    #fc1 = fc_layer("fc1", fc0, [128, NUM_LABELS])
    W_fc1 = weight_variable([512, NUM_LABELS])
    b_fc1 = bias_variable([NUM_LABELS])

    y_conv=tf.nn.softmax(tf.matmul(fc0_drop, W_fc1) + b_fc1)
     
    return y_conv #tf.reshape(y_conv, [data_shape[0],data_shape[1],data_shape[2]])


import os.path

def main():
    
    d2 = True
    num_channels = 3
    
    INIT_RATE = 0.001
    LR_DECAY_FACTOR = 0.1
    
    nr_epochs = 500
    
    num_rotations = 10
    patch_dim = 32
    #relSampling = 0.05
    #relRadius = 0.1
    #radius = pc_diameter*relRadius
    relL = 0.07
    
    BATCH_SIZE = 5
    
    fileName = '/media/hasan/DATA/Fac/BMC Master/Thesis/Models/bunny/reconstruction/plytest/bun_zipper.ply'
    reader = PlyReader.PlyReader()
    start_time = time.time()
    reader.read_ply(fileName)
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

        # Create input/output placeholder variables for the graph (to be filled manually with data)
        # Placeholders MUST be filled for each session.run()
        net_x = tf.placeholder("float", X.shape, name="in_x")
        net_y = tf.placeholder("float", Y.shape, name="in_y")
        
        
        #lr = tf.placeholder(tf.float32)
        # Build the graph that computes predictions and assert that network output is compatible
        
        output = build_graph(net_x, 0.5, reader.num_classes, d2=d2)
        
        print 'output shape: ',output.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
        print 'X shape: ',  net_x.get_shape().as_list()
        assert (output.get_shape().as_list() == net_y.get_shape().as_list() )
        
        global_step = tf.Variable(0, trainable=False)
        
        #compute cross entropy loss
        
        logy = tf.log(output + 1e-10)
        mult = net_y*logy
        cross_entropy = -tf.reduce_sum(mult)
        #tf.reshape(net_y, [net_y.get_shape().as_list()[1],net_y.get_shape().as_list()[0]])
        
        decay_steps = int(batches_per_epoch)
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INIT_RATE, global_step, decay_steps, LR_DECAY_FACTOR,  staircase=True)
        tf.scalar_summary('learning_rate', lr)
        opt = tf.train.AdamOptimizer(lr)
        #opt = tf.train.GradientDescentOptimizer(0.000001)
        train_op = opt.minimize(cross_entropy, global_step=global_step)
        
        
    
        correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(net_y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #train_op = opt.apply_gradients(opt.compute_gradients(total_loss), global_step=global_step)

        # Create initialization "op" and run it with our session 
        init = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        sess.run(init)
        
        # Create a saver and a summary op based on the tf-collection
        saver = tf.train.Saver(tf.all_variables())
        #summary_op = tf.merge_all_summaries()
        #summary_writer = tf.train.SummaryWriter('.', graph_def=sess.graph_def)
                
        #saver.restore(sess, 'model_1600_32.ckpt')   # Load previously trained weights
                
        
        for epoch in range(nr_epochs):
                print "Starting epoch ", epoch
                for batch in range(batches_per_epoch):
                    oldX=X
                    oldY=Y
                    X, Y= reader.next_batch(BATCH_SIZE, num_rotations=num_rotations, num_channels=num_channels, d2=d2)
                
                    start_time = time.time()
                    #res, absy_, logy_, mult_,err = sess.run([output,absy, logy, mult, cross_entropy], feed_dict={net_x:X, net_y: Y})
                    #_, error,summary = sess.run([train_op, cross_entropy, summary_op], feed_dict={net_x:X, net_y: Y})
                    _, error = sess.run([train_op, cross_entropy], feed_dict={net_x:X, net_y: Y})
                    duration = time.time() - start_time
                    print "Batch:", batch ,"  Loss: ", error, "   Duration (sec): ", duration

                    #summary_writer.add_summary(summary, batch)
                    
                    #Xv, Yv = reader.next_batch_val(BATCH_SIZE)
                    #acc = sess.run(accuracy, feed_dict={net_x:Xv, net_y: Yv})
                    #print "Batch:", batch ,"  Loss: ", error, " Test accuracy: ", acc, "   Duration (sec): ", duration
                    # Save the model checkpoint periodically.
                    if batch % 100 == 0:
                        save_path = saver.save(sess, "model_1600_32_1.ckpt")
                        print "Model saved in file: ",save_path




    print 'done'
if __name__ == "__main__":
    main()


