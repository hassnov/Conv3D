#!/usr/bin/env python

import time
import numpy
import math
import tensorflow as tf
import cluster_points_fpfh
import cluster_points_parallel
import os.path
import autoencoderutils
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
    
    num_rotations_test = 40
    BATCH_SIZE_TEST = 5
    
    BATCH_SIZE_FULL = num_rotations_test*BATCH_SIZE_TEST
    BATCH_SIZE_FULL = 10
    
    patch_dim = 32
    start_time = time.time()
    #dir_temp = "temp_32/"
    dir_temp = "../temp_5nr/"
    #dir_test = "temp_87/"
    reader = cluster_points_parallel.ClusterPoints()
    
    reader.load_dataset_and_labels(dir_temp=dir_temp, k=300, permutation_file="permutation_straight")
    #reader.permutation_num = 1800000

    
    reader_test = cluster_points_fpfh.ClusterPoints()
    reader_test.num_samples = 400
    reader_test.labels = numpy.arange(reader_test.num_samples)
    reader_test.patch_dim = 32

    
    samples_count = reader.compute_total_samples(num_aug)
    batches_per_epoch = samples_count/BATCH_SIZE_FULL
    print "Batches per epoch:", batches_per_epoch

    xShape = [BATCH_SIZE_FULL, patch_dim, patch_dim, patch_dim, 1]
    yShape = [BATCH_SIZE_FULL,]
    
    with tf.Graph().as_default() as graph:

        # Create input/output placeholder variables for the graph (to be filled manually with data)
        # Placeholders MUST be filled for each session.run()
        net_x = tf.placeholder("float", xShape, name="aein_x")
        
        
        recon, code = autoencoderutils.autoencoder_3_3_3_3_3(net_x, 32)
        assert (recon.get_shape().as_list() == net_x.get_shape().as_list())
        
        recon_loss = tf.nn.l2_loss((recon-net_x)/(BATCH_SIZE_FULL), name='ael2_loss')
        tf.add_to_collection('losses', recon_loss)
        total_loss = tf.add_n(tf.get_collection('losses'), name='aetotal_loss')
        #total_loss = recon_loss        
        global_step = tf.Variable(0, trainable=False)
        
        opt = tf.train.AdamOptimizer(0.00001)
        #opt = tf.train.GradientDescentOptimizer(0.001)
        train_op = opt.minimize(total_loss, global_step=global_step)

        # Create initialization "op" and run it with our session 
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.10)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
        sess.run(init)
        
        # Create a saver and a summary op based on the tf-collection
        saver = tf.train.Saver(tf.all_variables())

        saver.restore(sess, os.path.join(models_dir,'5models_ae32_720aug_5_5_3_5460_noise_relR_nr.ckpt'))   # Load previously trained weights
        
        #patch_num = 0
        #aug_num = 0   
        for epoch in range(1000):
                print "Starting epoch ", epoch
                for batch in range(batches_per_epoch):
                    start_time = time.time() 
                    #X, _= reader.next_batch_3d_file(BATCH_SIZE, num_rotations=num_aug, dir_temp=dir_temp)
                    X, _, augs, ids= reader.next_batch_3d_file_random(BATCH_SIZE_FULL, num_aug, dir_temp=dir_temp)
                    #print "ids:", ids
                    patch_time = time.time() - start_time 
                    #plotutils.plot_patch_3D(X[0])
                    #plt.show()
                    
                    #_, error, gstep, X_, code_ = sess.run([train_op, total_loss, global_step, recon, code], feed_dict={net_x:X})
                    error, gstep, X_, code_ = sess.run([total_loss, global_step, recon, code], feed_dict={net_x:X})
                    
                    X_[X_ > 0.5] = 1
                    X_[X_ <= 0.5] = 0
                    diff = numpy.abs(X - X_)
                    acc = numpy.sum(diff[diff > 0])
                    acc = 1 - (acc / (BATCH_SIZE_FULL * math.pow(reader.patch_dim, 3)))
                    
                    #for aug_i, patch in enumerate(code_):
                    #    numpy.save(dir_temp + "code/sample_" + str(ids[aug_i]) + "_" + str(augs[aug_i]), patch)
                    #print "ids", ids
                    #print "augs", augs
                    if reader.permutation_num < 1000:
                        print "Exiting when reader.permutation_num: ", reader.permutation_num
                        return 0
                    
                    """
                    if patch_num < reader.num_samples:
                        for patch in X_:
                            numpy.save(dir_temp + "recon/sample_" + str(patch_num) + "_" + str(aug_num), patch)
                            aug_num += 1
                            if aug_num == num_rotations:
                                aug_num = 0
                                patch_num +=1
                       """  
                        
                    duration = time.time() - start_time
                    #print "Batch:", batch ,"  Loss: {0:.6f}".format(error), "  acc: {0:.4f}".format(acc), "    global step: ", gstep, " sample: ", reader.sample_class_current,  "    Duration (sec): {0:.2f}".format(duration)
                    print "Batch:", batch ,"  Loss: {0:.6f}".format(error), "  acc: {0:.4f}".format(acc), "    global step: ", gstep, " sample: ", reader.permutation_num,  "  patch time: {0:.2f}  Duration (sec): {1:.2f}".format(patch_time, duration)
                        
                    if gstep % 1000 == 998:
                        start_time = time.time()
                        save_path = saver.save(sess, os.path.join(models_dir, "5models_ae32_720aug_5_5_3_5460_noise_relR_nr.ckpt"))
                        save_time = time.time() - start_time
                        print "Model saved in file: ",save_path, " in {0:.2f} seconds".format(save_time)
                        
                    # Evaluation
                    if gstep % 2000== 11999:
                        acc_test = []
                        for ii in range(10):
                            X_test, _= reader_test.next_batch_3d_file(BATCH_SIZE_TEST, num_rotations=num_rotations_test, dir_temp=dir_test)
                            error, _, X_ = sess.run([total_loss, global_step, recon], feed_dict={net_x:X_test})
                            X_[X_ > 0.5] = 1
                            X_[X_ <= 0.5] = 0
                            diff = numpy.abs(X_test - X_)
                            acc = numpy.sum(diff[diff > 0])
                            acc = 1 - (acc / (num_rotations_test * BATCH_SIZE_TEST * math.pow(reader.patch_dim, 3)))
                            acc_test.append(acc)
                        acc_test_mean = numpy.mean(acc_test)
                        print ".... test accuracy: ", acc_test_mean
                        print "index: ", reader_test.sample_class_current
                        with open("test_accuracy.txt", "a") as myfile:
                            myfile.write("Test accuracy : {0:.4f}    global step: {1}".format(acc_test_mean, int(gstep)) + '\n')
                            myfile.close()
                            

    print 'done'
if __name__ == "__main__":
    main()


