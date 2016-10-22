#!/usr/bin/env python

import time
import math
import numpy

import tensorflow as tf
import cluster_points_fpfh
import utils
import os.path
import convnnutils
from sampling import SampleAlgorithm
import autoencoderutils
#import plotutils
#import matplotlib.pyplot as plt

def main():    
    
    models_dir = '/home/titan/hasan/tr_models/'
    
    dir1 = os.path.dirname(os.path.realpath(__file__))
    num_rotations = 40
    BATCH_SIZE = 10
    patch_dim = 32
    start_time = time.time()
    dir_temp = "temp_32/"
    reader = cluster_points_fpfh.ClusterPoints()
    
    reader.load_dataset(dir_temp=dir_temp, k=300)
    
    samples_count = reader.compute_total_samples(num_rotations)
    batches_per_epoch = samples_count/BATCH_SIZE
    print "Batches per epoch:", batches_per_epoch

    xShape = [BATCH_SIZE*num_rotations, patch_dim, patch_dim, patch_dim, 1]
    yShape = [BATCH_SIZE*num_rotations,]
    
    with tf.Graph().as_default() as graph:

        # Create input/output placeholder variables for the graph (to be filled manually with data)
        # Placeholders MUST be filled for each session.run()
        net_x = tf.placeholder("float", xShape, name="in_x")
        #net_y = tf.placeholder(tf.int64, yShape, name="in_y")
        
        #logits, regularizers = convnnutils.build_graph_3d_7_5_3_nopool(net_x, 0.5, reader.num_clusters, train=True)
        
        recon = autoencoderutils.autoencoder_3_3_3(net_x, 128*2*2)
        assert (recon.get_shape().as_list() == net_x.get_shape().as_list())
        
        recon_loss = tf.nn.l2_loss((recon-net_x)/BATCH_SIZE, name='l2_loss')
        tf.add_to_collection('losses', recon_loss)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        #total_loss = recon_loss        
        global_step = tf.Variable(0, trainable=False)
        
        opt = tf.train.AdamOptimizer(0.00001)
        #opt = tf.train.GradientDescentOptimizer(0.001)
        train_op = opt.minimize(total_loss, global_step=global_step)

        # Create initialization "op" and run it with our session 
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
        sess.run(init)
        
        # Create a saver and a summary op based on the tf-collection
        saver = tf.train.Saver(tf.all_variables())

        saver.restore(sess, os.path.join(models_dir,'autoencoder5_5_32models_40_5_5_3_3_3443_fpfh_kmeans.ckpt'))   # Load previously trained weights
           
        for epoch in range(1000):
                print "Starting epoch ", epoch
                for batch in range(batches_per_epoch):
                    start_time = time.time() 
                    X, _= reader.next_batch_3d_file(BATCH_SIZE, num_rotations=num_rotations, dir_temp=dir_temp)
                    patch_time = time.time() - start_time
                    #plotutils.plot_patch_3D(X[0])
                    #plt.show()
                    
                    _, error, gstep = sess.run([train_op, total_loss, global_step], feed_dict={net_x:X})
                    duration = time.time() - start_time
                    print "Batch:", batch ,"  Loss: {0:.6f}".format(error), "    global step: ", gstep, " sample: ", reader.sample_class_current,  "    Duration (sec): {0:.2f}".format(duration)
                    
                    if batch % 1000 == 999:
                        start_time = time.time()
                        save_path = saver.save(sess, os.path.join(models_dir, "autoencoder5_5_32models_40_5_5_3_3_3443_fpfh_kmeans.ckpt"))
                        save_time = time.time() - start_time
                        print "Model saved in file: ",save_path, " in {0:.2f} seconds".format(save_time)

    print 'done'
if __name__ == "__main__":
    main()


