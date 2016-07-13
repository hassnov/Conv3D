#!/usr/bin/env python

import time
import tensorflow as tf
import os.path
import convnnutils
import dataCreator
import numpy
import PlyReader
import utils

def main():
        
    nr_epochs = 500
    dir1 = os.path.dirname(os.path.realpath(__file__))
    
    num_rotations=5
    BATCH_SIZE=1
    num_samples = 5
    patch_dim = 8
    relL = 0.07
    dir1 = os.path.dirname(os.path.realpath(__file__))
    fileName = os.path.join(dir1, 'plytest/bun_zipper.ply')
    reader = PlyReader.PlyReader()
    reader_noise = PlyReader.PlyReader()
    
    reader.read_ply(fileName, num_samples=num_samples, add_noise=False, noise_prob=0.2, noise_factor=0.02)
    pc_diameter = utils.get_pc_diameter(reader.data)
    l = relL*pc_diameter
    reader.set_variables(l=l, patch_dim=patch_dim)
    
    reader_noise.read_ply(fileName, num_samples=num_samples, add_noise=True, noise_prob=0.2, noise_factor=0.02)
    reader_noise.set_variables(l=l, patch_dim=patch_dim)
    
    samples_count = reader.compute_total_samples(num_rotations)
    batches_per_epoch = samples_count/BATCH_SIZE
    print "Batches per epoch:", batches_per_epoch
    batches_per_epoch = 25

    start_time = time.time()
    X, Y = reader.patch_3d( reader.data[0,...] )
    
    #X = numpy.load('data/train_data_' + str(0) +'.npy')
    #Y = numpy.load('data/train_label_' + str(0) +'.npy')
    print 'batch time: ', time.time() - start_time
    
    
    with tf.Graph().as_default() as graph:


        net_x = tf.placeholder("float", X.shape, name="in_x")
        net_y = tf.placeholder(tf.int64, Y.shape, name="in_y")
        
        logits, regularizers, conv1, pool1, h_fc0 = convnnutils.build_graph_3_3_512(net_x, 0.5, reader.num_classes, train=False)
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, net_y))
        loss += 5e-4 * regularizers
        
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
        #summary_op = tf.merge_all_summaries()
        #summary_writer = tf.train.SummaryWriter('.', graph_def=sess.graph_def)
        ckpt = tf.train.get_checkpoint_state(dir1)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)   # Load previously trained weights
            print 'model restored: ' + ckpt.model_checkpoint_path
        else:
            print 'no model to restore'
        
        print [v.name for v in tf.all_variables()]
        b = 0
        
        c1_1s = numpy.zeros((reader.data.shape[0], 8 * 8 * 8 * 32))
        p1_1s = numpy.zeros((reader.data.shape[0], 4 * 4 * 4 * 32))
        f0_1s = numpy.zeros((reader.data.shape[0], 512))
        
        c1_2s = numpy.zeros((reader.data.shape[0], 8 * 8 * 8 * 32))
        p1_2s = numpy.zeros((reader.data.shape[0], 4 * 4 * 4 * 32))
        f0_2s = numpy.zeros((reader.data.shape[0], 512))
        
        #for b in range(reader.data.shape[0]):
        for b in range(100):
            #X, Y= reader.next_batch(BATCH_SIZE, num_rotations=num_rotations)
            X, Y= reader.patch_3d(reader.data[b])
            start_time = time.time()
            X2, Y2 = reader_noise.patch_3d(reader.data[b])
            
            
            c1_1, p1_1, f0_1 = sess.run([conv1, pool1, h_fc0], feed_dict={net_x:X, net_y: Y})
            c1_1s[b] = numpy.reshape(c1_1, (8 * 8 * 8 * 32))
            p1_1s[b] = numpy.reshape(p1_1, (4 * 4 * 4 * 32))
            f0_1s[b] = numpy.reshape(f0_1, (512))
            duration = time.time() - start_time
            
            c1_2, p1_2, f0_2 = sess.run([conv1, pool1, h_fc0], feed_dict={net_x:X2, net_y: Y2})
            c1_2s[b] = numpy.reshape(c1_2, (8 * 8 * 8 * 32))
            p1_2s[b] = numpy.reshape(p1_2, (4 * 4 * 4 * 32))
            f0_2s[b] = numpy.reshape(f0_2, (512))
            
            
            #if b % 50 == 0:
            print "point:", b, "   Duration (sec): ", duration#, "    loss:    ", error, "    Accuracy: ", acc #, "   Duration (sec): ", duration

    print 'done'
if __name__ == "__main__":
    main()
    
    
