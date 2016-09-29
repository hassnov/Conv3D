#!/usr/bin/env python

import time
import tensorflow as tf
import os.path
import convnnutils
import dataCreator
import numpy

def main():
        
    nr_epochs = 500
    dir1 = os.path.dirname(os.path.realpath(__file__))
    
    num_rotations = 20
    BATCH_SIZE = 4
    reader = dataCreator.create_reader()
    samples_count = reader.compute_total_samples(num_rotations)
    batches_per_epoch = samples_count/BATCH_SIZE
    print "Batches per epoch:", batches_per_epoch
    #batches_per_epoch = 25
    #return 0
    start_time = time.time()
    X,Y = reader.next_batch(BATCH_SIZE, num_rotations=num_rotations)
    #X = numpy.load('data/train_data_' + str(0) +'.npy')
    #Y = numpy.load('data/train_label_' + str(0) +'.npy')
    print 'batch time: ', time.time() - start_time
    
    
    with tf.Graph().as_default() as graph:


        net_x = tf.placeholder("float", X.shape, name="in_x")
        net_y = tf.placeholder(tf.int64, Y.shape, name="in_y")
        
        logits, regularizers,_,_,_,_ = convnnutils.build_graph_3d(net_x, 0.5, reader.num_samples, train=False)
        
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, net_y))
        loss += 5e-4 * regularizers
        
        print 'logits shape: ',logits.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
        print 'X shape: ',  net_x.get_shape().as_list()
        
        global_step = tf.Variable(0, trainable=False)
        
        correct_prediction = tf.equal(tf.argmax(logits,1), net_y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Create initialization "op" and run it with our session 
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
        sess.run(init)
        
        # Create a saver and a summary op based on the tf-collection
        saver = tf.train.Saver(tf.all_variables())
        #summary_op = tf.merge_all_summaries()
        #summary_writer = tf.train.SummaryWriter('.', graph_def=sess.graph_def)
        """ckpt = tf.train.get_checkpoint_state(dir1)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)   # Load previously trained weights
            print 'model restored: ' + ckpt.model_checkpoint_path
        else:
            print 'no model to restore'
        """
        saver.restore(sess, 'bunny_4_10_100.ckpt')   # Load previously trained weights
        print [v.name for v in tf.all_variables()]
        b = 0

        accs = []
        for epoch in range(nr_epochs):
                print "Starting epoch ", epoch
                for batch in range(batches_per_epoch):
                    X, Y= reader.next_batch(BATCH_SIZE, num_rotations=num_rotations)
                    #print 'Xshape: ' , X.shape, '	Yshape: ', Y.shape, '	net_x: ', net_x.get_shape().as_list()
                    """
                    #X = numpy.load('data/test_data_' + str(b) +'.npy')
                    #Y = numpy.load('data/test_label_' + str(b) +'.npy')
                    for bb in range(3):
                        XX = numpy.load('data/test_data_' + str((1 + b + bb) % 5) +'.npy')
                        YY = numpy.load('data/test_label_' + str((1 + b + bb) % 5) +'.npy')
                        Y = numpy.hstack((YY, Y))
                        X = numpy.vstack((XX, X))
                    b = (b + 1) % 5"""
                    start_time = time.time()
                    error, acc = sess.run([loss, accuracy], feed_dict={net_x:X, net_y: Y})
                    duration = time.time() - start_time
                    print "Batch:", batch, "	loss:	", error, "	Accuracy: ", acc #, "   Duration (sec): ", duration
                    accs.append(acc)
                    if batch % 20 == 0:
                        print 'mean acc: ', numpy.mean(accs)
                        #acc = []

    print 'done'
if __name__ == "__main__":
    main()
    
    
