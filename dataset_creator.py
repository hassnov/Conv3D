import os.path
import numpy as np
#import plotutils
#from mayavi import mlab
from sampling import SampleAlgorithm
import time
import cluster_points

patch_dim = 32
         
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
    
    reader.create_dataset(dir_temp='temp1/')
    
    
    return 0

    

def load(dir_temp='temp/'):
    reader = cluster_points.ClusterPoints()
    reader.load_dataset(dir_temp='temp/')
    
    
#load(dir_temp='temp1/', bandwidth=0.0001)
main()

