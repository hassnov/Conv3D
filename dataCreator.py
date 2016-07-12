import time
import PlyReader
import utils
import os.path
import numpy as np



def create_reader():
    
    
    
    num_samples = 2
    patch_dim = 14
    #relSampling = 0.05
    #relRadius = 0.1
    #radius = pc_diameter*relRadius
    relL = 0.07
    
    dir1 = os.path.dirname(os.path.realpath(__file__))
        
    fileName = os.path.join(dir1, 'plytest/bun_zipper.ply')
    reader = PlyReader.PlyReader()
    start_time = time.time()
    reader.read_ply(fileName, num_samples=num_samples)
    print 'reading time: ', time.time() - start_time
    pc_diameter = utils.get_pc_diameter(reader.data)
    l = relL*pc_diameter
    print 'classes: ', reader.num_classes
    reader.set_variables(l=l, patch_dim=patch_dim)
    
    return reader


def create_data():
    
    num_rotations = 20
    BATCH_SIZE = 10
    reader = create_reader()
    ratio = 0.8
    epsilon = 0.00001
    samples_count = reader.compute_total_samples(num_rotations)
    X, Y= reader.next_batch(BATCH_SIZE, num_rotations=num_rotations)
    train_data = np.zeros((int(X.shape[0]*ratio), 32, 32, 32, 1))
    train_labels = np.zeros((int(X.shape[0]*ratio),))
    test_data = np.zeros((int(X.shape[0]*(1 - ratio + epsilon)), 32, 32, 32, 1))
    test_labels = np.zeros((int(X.shape[0]*(1 - ratio + epsilon)),))
    
    
    print 'train_data shape: ', train_data.shape
    print 'train_labels shape: ', train_labels.shape
    print 'test_data shape: ', test_data.shape
    print 'test_labels shape: ', test_labels.shape
    start_time = time.time()
    
    for i in range(samples_count // num_rotations):
        X, Y= reader.next_batch(BATCH_SIZE, num_rotations=num_rotations)
        ii = np.random.permutation(Y.shape[0])
        X = X[ii]
        Y = Y[ii]
        """train_data =    np.vstack((X[0:int(X.shape[0]*0.8), ...], train_data))
        train_labels =  np.vstack((Y[0:int(X.shape[0]*0.8), ...], train_labels))
        test_data =     np.vstack((X[int(X.shape[0]*0.8): , ...], test_data)) 
        test_labels =   np.vstack((Y[int(X.shape[0]*0.8): , ...], test_labels))"""
        np.save('train_data_' + str(i), X[0:int(X.shape[0]*0.8), ...])
        np.save('train_label_' + str(i), Y[0:int(X.shape[0]*0.8), ...])
        np.save('test_data_' + str(i), X[int(X.shape[0]*0.8):, ...])
        np.save('test_label_' + str(i), Y[int(X.shape[0]*0.8):, ...])
    
    """ tr_data = np.asarray(train_data)
    tr_lables = np.asarray(train_labels)
    tst_data = np.asarray(test_data)
    tst_labels = np.asarray(test_labels)
        """
    print 'data time: ', time.time() - start_time
    return train_data, train_labels, test_data, test_labels
