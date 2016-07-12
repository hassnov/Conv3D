#!/usr/bin/env python

import time
import math
import numpy
import tensorflow as tf
import PlyReader
import utils
import os.path
import convnnutils


def main():
    
    d2 = False
    num_channels = 1
    
    INIT_RATE = 0.00001
    LR_DECAY_FACTOR = 0.1
    
    nr_epochs = 500
    
    num_rotations = 5
    patch_dim = 32
    #relSampling = 0.05
    #relRadius = 0.1
    #radius = pc_diameter*relRadius
    relL = 0.07
    
    BATCH_SIZE = 10
    
    dir1 = os.path.dirname(os.path.realpath(__file__))
        
    fileName = os.path.join(dir1, 'plytest/bun_zipper.ply')
    reader = PlyReader.PlyReader()
    start_time = time.time()
    reader.read_ply(fileName, num_samples=3)
    print 'reading time: ', time.time() - start_time
    pc_diameter = utils.get_pc_diameter(reader.data)
    l = relL*pc_diameter
    print 'classes: ', reader.num_classes
    reader.set_variables(l=l, patch_dim=patch_dim)
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
        net_y = tf.placeholder(tf.int64, Y.shape, name="in_y")
        
        logits, regularizers = convnnutils.build_graph_3d(net_x, 0.5, reader.num_classes, train=True)
        
        print 'logits shape: ',logits.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
        print 'X shape: ',  net_x.get_shape().as_list()
        print 'press key to start train...'
        wait()
        global_step = tf.Variable(0, trainable=False)
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, net_y))
        loss += 5e-4 * regularizers
        
        decay_steps = int(batches_per_epoch)
        lr = tf.train.exponential_decay(INIT_RATE, global_step, decay_steps, LR_DECAY_FACTOR,  staircase=True)
        opt = tf.train.AdamOptimizer(0.001)
        #opt = tf.train.GradientDescentOptimizer(0.5)
        train_op = opt.minimize(loss, global_step=global_step)
        
        correct_prediction = tf.equal(tf.argmax(logits,1), net_y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Create initialization "op" and run it with our session 
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
        sess.run(init)
        
        # Create a saver and a summary op based on the tf-collection
        saver = tf.train.Saver(tf.all_variables())
        
        ckpt = tf.train.get_checkpoint_state(dir1)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)   # Load previously trained weights
            print 'model restored: ' + ckpt.model_checkpoint_path
        else:
            print 'no model to restore'
        
        print [v.name for v in tf.all_variables()]
        
        #saver.restore(sess, 'model_bunny_5_10.ckpt')   # Load previously trained weights
                        
        #for epoch in range(nr_epochs):
        for epoch in range(1):
                print "Starting epoch ", epoch
                for batch in range(batches_per_epoch):
                    X, Y= reader.next_batch(BATCH_SIZE, num_rotations=num_rotations, num_channels=num_channels, d2=d2)
                
                    start_time = time.time()
                    _, error, acc, gstep = sess.run([train_op, loss, accuracy, global_step], feed_dict={net_x:X, net_y: Y})
                    duration = time.time() - start_time
                    print "Batch:", batch ,"  Loss: ", error, "    Accuracy: ", acc, "    global step: ", gstep#, "   Duration (sec): ", duration

                    if batch % 100 == 0:
                        save_path = saver.save(sess, "bunny_10_5_1000.ckpt", global_step=global_step)
                        print "Model saved in file: ",save_path
                    
                    if batch % 300 == 0:
                        break
        
        print 'start testing.....'
        
        for epoch in range(nr_epochs):
                print "Starting epoch ", epoch
                for batch in range(batches_per_epoch):
                    X, Y= reader.next_batch(BATCH_SIZE, num_rotations=num_rotations, num_channels=num_channels, d2=d2)
                
                    start_time = time.time()
                    acc, gstep = sess.run([accuracy, global_step], feed_dict={net_x:X, net_y: Y})
                    duration = time.time() - start_time
                    print "Accuracy: ", acc , "  global step: ", gstep, "  Duration (sec): ", duration

    print 'done'
if __name__ == "__main__":
    main()


