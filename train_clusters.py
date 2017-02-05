#!/usr/bin/env python

import time
import math
import numpy

import tensorflow as tf
import cluster_points_parallel2
import utils
import os.path
import convnnutils
from sampling import SampleAlgorithm
#import plotutils
#import matplotlib.pyplot as plt

def main():    
    
    models_dir = '/home/hasan/hasan/tr_models/'
    
    dir1 = os.path.dirname(os.path.realpath(__file__))
    num_rotation = 90
    num_noise = 2
    num_relR = 2
    nrs = 2
    num_aug = num_rotation*num_noise*num_relR*nrs
    BATCH_SIZE = 1
    
    BATCH_SIZE_FULL = BATCH_SIZE * num_aug
    
    patch_dim = 32
    BATCH_SIZE_FULL = 50
    #files_file = os.path.join(dir1, 'files_file.txt')
    start_time = time.time()
    dir_temp = "../temp_5nr/"
    reader = cluster_points_parallel2.ClusterPoints() 
    reader.load_dataset_and_labels(dir_temp=dir_temp, k=201, create_labels=False)
    
    
    samples_count = reader.compute_total_samples(num_aug)
    batches_per_epoch = samples_count/BATCH_SIZE_FULL
    print "Batches per epoch:", batches_per_epoch
    

    
    #start_time = time.time()
    #X,Y = reader.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations)

    xShape = [BATCH_SIZE_FULL, patch_dim, patch_dim, patch_dim, 1]
    yShape = [BATCH_SIZE_FULL,]
    
    #print 'batch time: ', time.time() - start_time
    
    
    with tf.Graph().as_default() as graph:

        # Create input/output placeholder variables for the graph (to be filled manually with data)
        # Placeholders MUST be filled for each session.run()
        net_x = tf.placeholder("float", xShape, name="in_x")
        net_y = tf.placeholder(tf.int64, yShape, name="in_y")
        
        #lr = tf.placeholder(tf.float32)
        # Build the graph that computes predictions and assert that network output is compatible
        
        #logits, regularizers = convnnutils.build_graph_3d_5_5_3_3_3_4000(net_x, 0.5, reader.num_clusters, train=True, wd=5e-4)
        logits, regularizers = convnnutils.build_graph_3d_5_5_3_3_3(net_x, 0.5, reader.num_clusters, train=True, wd=5e-4)
        
        print 'logits shape: ',logits.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
        print 'X shape: ',  net_x.get_shape().as_list()
        #assert (output.get_shape().as_list() == net_y.get_shape().as_list() )
        
        global_step = tf.Variable(0, trainable=False)
        
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, net_y))
        tf.add_to_collection('losses', cross_entropy)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
        
        output = tf.nn.relu(logits)
        #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.clip_by_value(net_y,1e-10,1.0)))
        #loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
        
        
        decay_steps = int(batches_per_epoch)
        #lr = tf.train.exponential_decay(INIT_RATE, global_step, decay_steps, LR_DECAY_FACTOR,  staircase=True)
        opt = tf.train.AdamOptimizer(0.0001)
        #opt = tf.train.GradientDescentOptimizer(0.001)
        train_op = opt.minimize(loss, global_step=global_step)
        
        correct_prediction = tf.equal(tf.argmax(logits,1), net_y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Create initialization "op" and run it with our session 
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)
        #gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
        sess.run(init)
        
        # Create a saver and a summary op based on the tf-collection
        #saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=2)
        
        
        #saver.restore(sess, os.path.join(models_dir,'5models_720aug_5_5_3_3_3_3057_90rots_noise_relR_nr_100clusters_fpfh.ckpt'))   # Load previously trained weights
        #saver.restore(sess, os.path.join(models_dir,'5models_720aug_5_5_3_3_3_3057_90rots_noise_relR_nr_108clusters_ae.ckpt'))   # Load previously trained weights
        #saver.restore(sess, os.path.join(models_dir,'9models_5400aug_5_5_3_3_3_5460_90rots_noise_relR_nr_400fpfh.ckpt'))   # Load previously trained weights
        #saver.restore(sess, os.path.join(models_dir,'9models_5400aug_5_5_3_3_3_1024desc_5460_90rots_noise_relR_nr_400fpfh.ckpt'))   # Load previously trained weights
        #saver.restore(sess, os.path.join(models_dir,'1models_8aug_5_5_3_3_3_537_1rots_noise_relR_nr.ckpt'))   # Load previously trained weights
        #saver.restore(sess, os.path.join(models_dir,'1models_360aug_5_5_3_3_3_537_90rots_relR_nr.ckpt'))   # Load previously trained weights
        #saver.restore(sess, os.path.join(models_dir,'1models_360aug_5_5_3_3_3_537_90rots_noise_nr.ckpt'))   # Load previously trained weights
        #saver.restore(sess, os.path.join(models_dir,'1models_360aug_5_5_3_3_3_537_90rots_noise_relR.ckpt'))   # Load previously trained weights
        #saver.restore(sess, os.path.join(models_dir,'1models_720aug_5_5_3_3_3_537_full.ckpt'))   # Load previously trained weights
        #saver.restore(sess, os.path.join(models_dir,'5models_720aug_5_5_3_3_3_3057_90rots_noise_relR_nr_201clusters_ae.ckpt'))   # Load previously trained weights
        
        
        for epoch in range(1000):
                print "Starting epoch ", epoch
                for batch in range(batches_per_epoch):
                    start_time = time.time() 
                    #X, Y= reader.next_batch_3d_file(BATCH_SIZE, num_rotations=num_aug, dir_temp=dir_temp)
                    X, Y, _, _= reader.next_batch_3d_file_random_train(BATCH_SIZE_FULL, num_aug, dir_temp=dir_temp)
                    
                    patch_time = time.time() - start_time
                    #plotutils.plot_patch_3D(X[0])
                    #plt.show()
                    
                    _, error, acc, gstep = sess.run([train_op, loss, accuracy, global_step], feed_dict={net_x:X, net_y: Y})
                    duration = time.time() - start_time
                    #print "Batch:", batch ,"  Loss: {0:.6f}".format(error), "    Accuracy: {0:.4f}".format(acc), "    global step: ", gstep, "sample: ", reader.sample_class_current,  "    Duration (sec): {0:.2f}".format(duration)
                    print "Batch:", batch ,"  Loss: {0:.6f}".format(error), "  acc: {0:.4f}".format(acc), "    global step: ", gstep, " sample: ", reader.permutation_num_train,  "  patch time: {0:.2f}  Duration (sec): {1:.2f}".format(patch_time, duration)
                    #print output1

                    if gstep % 1000 == 10:
                        start_time = time.time()
                        X, Y, _, _= reader.next_batch_3d_file_random_test(BATCH_SIZE_FULL, num_aug, dir_temp=dir_temp)                        
                        error1, acc1, gstep = sess.run([loss, accuracy, global_step], feed_dict={net_x:X, net_y: Y})
                        print "test error: {0:.6}    test accuracy: {1:.4}".format(error1, acc1)
                        with open("test_error.txt", "a") as myfile:
                            myfile.write("train error: {0:.6f}, test error: {1:.6f}".format(error, error1) + '\n')
                            myfile.close()

                        with open("test_acc.txt", "a") as myfile:
                            myfile.write("train acc: {0:.6f}, test acc: {1:.6f}".format(acc, acc1) + '\n')
                            myfile.close()

                    if gstep % 1000 == 999:
                        start_time = time.time()
                        save_path = saver.save(sess, os.path.join(models_dir, "5models_720aug_5_5_3_3_3_3057_90rots_noise_relR_nr_201clusters_ae.ckpt"))
                        save_time = time.time() - start_time
                        print "Model saved in file: ",save_path, " in {0:.2f} seconds".format(save_time)

    print 'done'
if __name__ == "__main__":
    main()

