import numpy as np
import os.path
import time
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import cv2
from scipy import spatial
import pyflann
import sys

#import plotutils
#from mayavi import mlab


def cluster_list_kmeans(eigen_list, k):
    km = KMeans(k).fit(eigen_list)
    labels = km.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    num_clusters = n_clusters_
    print "kmeans clusters: ", num_clusters 
    return labels, num_clusters

def cluster_list_meanshift(eigen_list, bandwidth=0.00015):
        
        bandwidth = estimate_bandwidth(eigen_list, quantile=0.5)
        print "bandwidth: ", bandwidth
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(eigen_list)
        labels = ms.labels_
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        num_clusters = n_clusters_
        labels = ms.labels_
        print("number of estimated clusters : %d" % n_clusters_)
        return labels, num_clusters

def load_dataset(k, dir_temp='temp/', num_rotations=40, sample_class_start=0):
    
    #meta = np.load(dir_temp + "meta.npy")
    #num_samples = 3440 
    num_samples = np.load(dir_temp + "meta.npy")[1]
    print "num samples: ", num_samples
    #patch_dim = meta[2]
    #num_rotations = 40
    
    print "loading coded version..."
    if (False):
        codes = np.zeros((num_samples*num_rotations, 32))
        for i in range(num_samples):
            sample_id = sample_class_start + i
            t0 = time.time()
            for aug_num in range(num_rotations):
                if aug_num == num_rotations:
                    break
                temp_file_sample = dir_temp + "code/" + "sample_" + str(sample_id) + '_' + str(aug_num) + '.npy'
                if not os.path.isfile(temp_file_sample):
                    print temp_file_sample
                    assert(os.path.isfile(temp_file_sample))
                codes[i*num_rotations + aug_num] = np.load(temp_file_sample)
            #if i % 100 == 0:
            print "{0} in {1} seconds".format(sample_id, time.time()-t0)
            sys.stdout.flush()
        np.save(dir_temp + "codes", codes)
    codes = np.load(dir_temp + "codes.npy")
    print "clustering..."
    
    if os.path.isfile(dir_temp + "kmeans_result" + str(k) + ".npy"):
        result = np.load(dir_temp + "kmeans_result" + str(k) + ".npy")
    else:
        t0 = time.time()
        flann = pyflann.FLANN()
        result = flann.kmeans(codes, num_clusters=k, max_iterations=40)
        print "clustered in {0:.2} seconds".format(time.time() - t0)
        np.save(dir_temp + "kmeans_result" + str(k), np.asarray(result))
    
    print "get labels..."
    tree = spatial.KDTree(result)
    labels = []
    t0 = time.time()
    for point in codes:
        _, indices = tree.query(point, 1)
        labels.append(indices)
    print "labeled in {0:.2} seconds".format(time.time() - t0)
    np.save(dir_temp + "labels" + str(k), np.asarray(labels))


def dominant(arr, element):
    if not np.array([element]) in arr:
        return False
    list_unique = np.unique(arr)
    percantages = np.zeros(list_unique.shape)
    size = arr.shape[0]
    for i, e in enumerate(list_unique):
        percantages[i] = len(np.where(arr == e)[0]) / float(size)
    maxi = np.argmax(percantages)
    if list_unique[maxi] == element:
        return True
    return False
    
def merge_clusters(dir_temp, num_rotations, num_clusters):
    labels_orig = np.load(dir_temp + "labels" + str(num_clusters) +".npy")
    labels_res = np.asarray(labels_orig, np.int32)
    
    i = 0
    #for i in range(0, len(labels_orig), num_rotations):
    while i < len(labels_res):
        i1 = i - (i % num_rotations)
        if not dominant(labels_res[i1:i1+num_rotations], labels_res[i]):
            i +=1
            continue
        #if not labels_res[i:i+num_rotations] == np.full([num_rotations,], labels_res[i], np.int32):
        if not np.all(labels_res[i1:i1+num_rotations] == np.full([num_rotations,], labels_res[i1], np.int32)):
            curr_batch = np.unique(labels_res[i1:i1+num_rotations])
            for label in curr_batch:
                for j in np.where(labels_res == label)[0]:
                    if labels_res[j] != label:
                        continue
                    j1 = j - (j % num_rotations)
                    if dominant(labels_res[j1:j1+num_rotations], label):
                        labels_res[j1:j1+num_rotations] = np.full([num_rotations,], labels_res[i], np.int32)
        i = i1 + num_rotations
    np.save(dir_temp + "labels_res" + str(num_clusters) + ".npy", labels_res)
    
def reduce_labels(labels, num_rotations):
    """
    size = len(labels) / num_rotations
    labels_train = np.zeros((size,))
    for i in range(0, size):
        labels_train[i] = labels[i*num_rotations]
    """
    labels_train = labels[range(0, len(labels), num_rotations)]
    labels_train += np.max(labels_train)
    labels_unique = np.unique(labels_train)
    for i, label in enumerate(labels_unique):
        labels_train[labels_train == label] = i
    print labels_train
    return labels_train

n_c = 10000
num_rotations=360*2
dir_temp = "../temp_5nr/"
load_dataset(n_c, dir_temp=dir_temp, num_rotations=num_rotations)
print "merging..."
merge_clusters(dir_temp=dir_temp, num_rotations=num_rotations, num_clusters=n_c)


labels_res = np.load(dir_temp + "labels_res" + str(n_c) +".npy")
reduced = reduce_labels(labels_res, num_rotations)
np.save(dir_temp + "reduced_labels" + str(n_c), reduced)

