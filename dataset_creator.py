import ConfigParser
import os.path
import numpy as np
import utils
#import plotutils
#from mayavi import mlab
from plyfile import PlyData
from sampling import SampleAlgorithm
import PlyReader3
import tensorflow as tf
import convnnutils
import time
import sys
import cluster_points
from sklearn.cluster import MeanShift, estimate_bandwidth

patch_dim = 32

def dummy():
    num_rotations = 40
    batch_size = 2
    X = np.zeros((batch_size*  num_rotations, patch_dim, patch_dim, patch_dim, 1), np.int32)
    Y = np.zeros((batch_size*  num_rotations), np.int32)
    z_axis = np.array([0, 0, 1])
    point_number = 0
    for i in range(500, 800):
        start_time = time.time()
        ref_points = np.load("temp/sample_" + str(i) + ".npy")
        r = np.max ([np.max(ref_points[:, 0]), np.max(ref_points[:, 1])])
        rz = r / 2
        print "r=", r
        for aug_num, theta in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
            if aug_num == num_rotations:
                break
            rot2d = utils.angle_axis_to_rotation(theta, z_axis)
            rot_points = utils.transform_pc(ref_points, rot2d)

            for rot_pt in rot_points:
                                    
                x = int(((rot_pt[0] + r) / (2 * r))*(patch_dim - 1))
                y = int(((rot_pt[1] + r) / (2 * r))*(patch_dim - 1))
                z = int(((rot_pt[2] + rz) / (2 * rz))*(patch_dim - 1))

                if (z >= 0) and (z < patch_dim) and (y >= 0) and (y < patch_dim) and (x >= 0) and (x < patch_dim):
                    #patch[x, y, z] = 1
                    #X[point_number*num_rotations + aug_num, x + patch_dim * (y + patch_dim * z)] = 1
                    X[point_number*num_rotations + aug_num, x, y, z, 0] = 1
            #X[point_number*num_rotations + aug_num, :] = patch.reshape((np.power(patch_dim, 3),))
            #Y[point_number*num_rotations + aug_num] = sample_class_current % num_classes
        duration = time.time() - start_time
        print "duration: {0:.4f}".format(duration)
            
def main():

    dir1 = os.path.dirname(os.path.realpath(__file__))
    #num_rotations = 40
    #BATCH_SIZE = 2
    patch_dim = 32
    files_file = os.path.join(dir1, 'files_file.txt')
    start_time = time.time()
    
    
    reader = cluster_points.ClusterPoints()
    #reader = PlyReader3.PlyReader()
    reader.create_reader(files_file, 0,
                          add_noise=False, noise_std=0.1, sampling_algorithm=SampleAlgorithm.ISS_Detector,
                          num_classes=100, relL=0.07, patch_dim=patch_dim,
                          use_normals_pc=False, use_point_as_mean=False, flip_view_point=False)
    
    reader.create_dataset()
    
    
    return 0




main()
    
