import os.path
import numpy as np
#import plotutils
#from mayavi import mlab
from sampling import SampleAlgorithm
import time
import cluster_points
import cluster_points_fpfh

patch_dim = 32
         
def main():
    dir1 = os.path.dirname(os.path.realpath(__file__))
    #num_rotations = 40
    #BATCH_SIZE = 2
    patch_dim = 32
    files_file = os.path.join(dir1, 'files_file4.txt')
    start_time = time.time()
    
    
    reader = cluster_points_fpfh.ClusterPoints()
    #reader = PlyReader3.PlyReader()
    reader.create_reader(files_file, 0,
                          add_noise=False, noise_std=0.1, sampling_algorithm=SampleAlgorithm.ISS_Detector,
                          num_classes=-1, relL=0.07, patch_dim=patch_dim,
                          use_normals_pc=False, use_point_as_mean=False, flip_view_point=False)
    
    reader.create_dataset(dir_temp='temp_fpfh/')
    
    
    return 0

    

def load(dir_temp='temp/', bandwidth=0.00001):
    reader = cluster_points_fpfh.ClusterPoints()
    reader.load_dataset(dir_temp=dir_temp, bandwidth=bandwidth)
    
    
load(dir_temp='temp_fpfh/', bandwidth=0.01)
#main()

