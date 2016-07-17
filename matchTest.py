#!/usr/bin/env python

import time
import tensorflow as tf
import os.path
import convnnutils
import dataCreator
import numpy
import PlyReader
import utils
import cv2
from numpy import dtype
from scipy import spatial
import plotutils


    
def main():
        
    nr_epochs = 500
    dir1 = os.path.dirname(os.path.realpath(__file__))
    
    num_rotations=1
    BATCH_SIZE=5
    samples_per_batch = BATCH_SIZE * num_rotations
    num_samples = 100
    patch_dim = 32
    relL = 0.07
    dir1 = os.path.dirname(os.path.realpath(__file__))
    fileName = os.path.join(dir1, 'plytest/bun_zipper.ply')
    reader = PlyReader.PlyReader()
    reader_noise = PlyReader.PlyReader()
    
    reader.read_ply(fileName, num_samples=num_samples, add_noise=False)
    pc_diameter = utils.get_pc_diameter(reader.data)
    l = relL*pc_diameter
    reader.set_variables(l=l, patch_dim=patch_dim)
    
    reader_noise.read_ply(fileName, num_samples=num_samples, add_noise=True, noise_prob=0.2, noise_factor=0.02)
    reader_noise.set_variables(l=l, patch_dim=patch_dim)
    
    samples_count = reader.compute_total_samples(num_rotations)
    batches_per_epoch = samples_count/BATCH_SIZE
    print "Batches per epoch:", batches_per_epoch
    #batches_per_epoch = 25

    start_time = time.time()
    #X, Y= reader.next_batch(BATCH_SIZE, num_rotations=num_rotations)
    
    print 'batch time: ', time.time() - start_time
    
    
    with tf.Graph().as_default() as graph:
        #net_x = tf.placeholder("float", X.shape, name="in_x")
        #net_y = tf.placeholder(tf.int64, Y.shape, name="in_y")
        
        net_x = tf.placeholder("float", [samples_per_batch, patch_dim, patch_dim, patch_dim, 1], name="in_x")
        net_y = tf.placeholder(tf.int64, [samples_per_batch,], name="in_y")
        
        logits, regularizers, conv1, pool1, h_fc0, h_fc1 = convnnutils.build_graph_3d(net_x, 0.5, reader.num_classes, train=False)
        
        #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, net_y))
        #loss += 5e-4 * regularizers
        
        print 'logits shape: ',logits.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
        print 'X shape: ',  net_x.get_shape().as_list()
        
        global_step = tf.Variable(0, trainable=False)
        
        correct_prediction = tf.equal(tf.argmax(logits,1), net_y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Create initialization "op" and run it with our session 
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
        sess.run(init)
        
        # Create a saver and a summary op based on the tf-collection
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, 'bunny_4_10_100.ckpt')   # Load previously trained weights
        
        print [v.name for v in tf.all_variables()]
        b = 0
        
        c1_shape = conv1.get_shape().as_list()
        p1_shape = pool1.get_shape().as_list()
        f0_shape = h_fc0.get_shape().as_list()
        f1_shape = h_fc1.get_shape().as_list()
        c1_1s = numpy.zeros((reader.samples.shape[0], c1_shape[1] * c1_shape[2] * c1_shape[3] * c1_shape[4]), dtype=numpy.float32)
        p1_1s = numpy.zeros((reader.samples.shape[0], p1_shape[1] * p1_shape[2] * p1_shape[3] * p1_shape[4]), dtype=numpy.float32)
        f0_1s = numpy.zeros((reader.samples.shape[0], f0_shape[1]), dtype=numpy.float32)
        f1_1s = numpy.zeros((reader.samples.shape[0], f1_shape[1]), dtype=numpy.float32)
        
        c1_2s = numpy.zeros((reader.samples.shape[0], c1_shape[1] * c1_shape[2] * c1_shape[3] * c1_shape[4]), dtype=numpy.float32)
        p1_2s = numpy.zeros((reader.samples.shape[0], p1_shape[1] * p1_shape[2] * p1_shape[3] * p1_shape[4]), dtype=numpy.float32)
        f0_2s = numpy.zeros((reader.samples.shape[0], f0_shape[1]), dtype=numpy.float32)
        f1_2s = numpy.zeros((reader.samples.shape[0], f1_shape[1]), dtype=numpy.float32)
        
        #for b in range(100):
        for b in range(samples_count // BATCH_SIZE):
            start_time = time.time()
            X, Y= reader.next_batch(BATCH_SIZE, num_rotations=num_rotations)
            X2, Y2= reader_noise.next_batch(BATCH_SIZE, num_rotations=num_rotations)
            
            i = b*num_rotations*BATCH_SIZE
            i1 = (b + 1)*num_rotations*BATCH_SIZE
            c1_1, p1_1, f0_1, f1_1 = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X, net_y: Y})
            c1_1s[i:i1] = numpy.reshape(c1_1, (samples_per_batch,c1_1s.shape[1]))
            p1_1s[i:i1] = numpy.reshape(p1_1, (samples_per_batch, p1_1s.shape[1]))
            f0_1s[i:i1] = numpy.reshape(f0_1, (samples_per_batch, f0_1s.shape[1]))
            f1_1s[i:i1] = numpy.reshape(f1_1, (samples_per_batch, f1_1s.shape[1]))
            
            c1_2, p1_2, f0_2, f1_2 = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X2, net_y: Y2})
            c1_2s[i:i1] = numpy.reshape(c1_2, (samples_per_batch, c1_2s.shape[1]))
            p1_2s[i:i1] = numpy.reshape(p1_2, (samples_per_batch, p1_2s.shape[1]))
            f0_2s[i:i1] = numpy.reshape(f0_2, (samples_per_batch, f0_2s.shape[1]))
            f1_2s[i:i1] = numpy.reshape(f1_2, (samples_per_batch, f1_2s.shape[1]))
            duration = time.time() - start_time
            
            matches = utils.match_des(c1_1s[i:i1], c1_2s[i:i1])
            plotutils.show_matches(reader.data, reader_noise.data, reader.samples, reader_noise.samples, matches)
            
            #if b % 50 == 0:
            print "point:", b, "   Duration (sec): ", duration#, "    loss:    ", error, "    Accuracy: ", acc #, "   Duration (sec): ", duration
        matches = utils.match_des(c1_1s, c1_2s)
        plotutils.show_matches(reader.data, reader_noise.data, reader.samples, reader_noise.samples, matches)

    print 'done'
    
if __name__ == "__main__":
    main()
    
    
