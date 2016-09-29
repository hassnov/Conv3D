import numpy as np
from numpy import linalg as LA
from plyfile import PlyData
from scipy import spatial
import os
import os.path
from sampling import Sampler
from sampling import SampleAlgorithm
import utils
import time
import logging
import plotutils
from mayavi import mlab
import enum
from sklearn.cluster import MeanShift, estimate_bandwidth



class ClusterPoints:
    
    data = None
    samples = None
    sample_indices = None
    num_samples = None
    num_clusters = None
    index = 0
    tree = None
    
    l = None
    relL=0.07    
    num_rotations = None
    patch_dim = None
    use_normals_pc = None
    use_point_as_mean = None
    flip_view_point = None
    sigma = None
    sample_class_start = 0
    sample_class_current = 0
    sampling_algorithm = None
      
    add_noise = False
    noise_std=0.1
    
    #rotation of whole body
    rotation_axis=[0, 0, 1]
    rotation_angle=0
    
    
    files_file = None
    files_list = []
    file_index = 0

    #nr_count = 1

    
    def fill_files_list(self):
        with open(self.files_file, 'r') as f:
            content = f.readlines()
        for file1 in content:
            if os.path.isfile(file1.rstrip('\n')):
                self.files_list.append(file1.rstrip('\n'))
            else:
                #TODO if dir
                if os.path.isdir(file1.rstrip('\n')):
                    files = self.files_in_dir(file1.rstrip('\n'))
                    for f in files:
                        self.files_list.append(f)
    
    def files_in_dir(self, dir1):
        #os.listdir()
        onlyfiles = [os.path.join(dir1, f) for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
        return onlyfiles
       
    def next_file(self):
        self.file_index = (self.file_index + 1) % len(self.files_list)
        file_name = self.files_list[self.file_index]
        self.read_ply(file_name)
        print 'file: ', file_name
    
    def create_reader(self, files_file, sample_class_start=0, add_noise =False,
                  noise_std=0.1, sampling_algorithm=SampleAlgorithm.Uniform, 
                  rotation_axis=[0, 0, 1], rotation_angle=0, num_classes=2000, 
                  relL=0.07, patch_dim=32,
                  use_normals_pc=False, use_point_as_mean=False, flip_view_point=False, sigma=0.7071):
        self.files_file = files_file
        self.add_noise = add_noise
        self.rotation_axis = rotation_axis
        self.rotation_angle = rotation_angle
        self.noise_std = noise_std
        self.sample_class_start = sample_class_start
        self.sample_class_current = sample_class_start
        self.num_samples = num_classes
        self.sampling_algorithm = sampling_algorithm
        self.fill_files_list()
        print 'files list: ', self.files_list
        
        self.relL = relL
        self.patch_dim=patch_dim
        self.use_normals_pc=use_normals_pc
        self.use_point_as_mean=use_point_as_mean
        self.flip_view_point=flip_view_point
        self.sigma=sigma
        
        
        #file_name = self.files_list[self.file_index]
        #print 'file: ', file_name
        #self.read_ply(file_name)
        


    def read_ply(self, file_name):
        num_samples = self.num_samples // len(self.files_list)
        if self.file_index == len(self.files_list) - 1:
            num_samples = num_samples + (self.num_samples - (num_samples * len(self.files_list)))
        
        root, ext = os.path.splitext(file_name)
        if not os.path.isfile(root + ".npy"):
            ply = PlyData.read(file_name)
            vertex = ply['vertex']
            (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
            points = zip(x.ravel(), y.ravel(), z.ravel())
            np.save(root + ".npy", points)
        else:
            points = np.load(root + ".npy")
            
        if self.add_noise:
            self.data = utils.add_noise_normal(points, std=self.nois_std)
        else:
            self.data = np.asarray(points)
        
        pc_diameter = utils.get_pc_diameter(self.data)
        self.l = self.relL*pc_diameter
        
        rot = utils.angle_axis_to_rotation(self.rotation_angle, self.rotation_axis)
        self.data = utils.transform_pc(self.data, rot)
        
        #plotutils.show_pc(self.data)
        #mlab.show()
                
        #TODO: better sampling
        print "sampling file: ", file_name
        self.samples, self.sample_indices = Sampler.sample(self.data, -1, num_samples, file_name=file_name, sampling_algorithm=self.sampling_algorithm)
        self.samples = self.samples[0:num_samples]
        self.sample_indices = self.sample_indices[0:num_samples]
        
        self.tree = spatial.KDTree(self.data)
                
        logging.basicConfig(filename='example.log',level=logging.DEBUG)
        return self.data
    
    
    def compute_total_samples(self, num_rotations=20):
        return self.num_samples*num_rotations
        
    
    def get_ref_points(self, samplept, r):
        i = self.tree.query_ball_point(samplept[0:3], r=r)
        _, indices = self.tree.query(samplept[0:3], k=len(i))
        #query_time = time.time() - start_time
        #print "query time: {0:.2f}".format(query_time)
        local_points = self.data[indices]
        #if local_points.shape[0] > 1000:
        #    local_points, _ = Sampler.sample(local_points, -1, 1000, sampling_algorithm=self.sampling_algorithm)
        if len(i) <= 8:
            return []
        if self.use_point_as_mean:
            mu = samplept[0:3]
        else:
            #TODO: compute real mean
            mu = np.zeros(3)
            point_count = local_points.shape[0]
            mu[0] = np.sum(local_points[:, 0]) / point_count
            mu[1] = np.sum(local_points[:, 1]) / point_count
            mu[2] = np.sum(local_points[:, 2]) / point_count
        if (self.use_normals_pc) and (samplept.shape[0] == 6):
            nr = samplept[3:6]
            #TODO: get_weighted_normal
        else:# calc normals
            cov_mat = utils.build_cov(local_points, mean=mu)
            w, v = LA.eigh(cov_mat)
            min_id = np.argmin(w)
            #print 'eigenvalues: ', w
            nr1 = np.transpose(v[:, min_id])
            nr = utils.normalize(nr1)
            
        z_axis = np.array([0, 0, 1])
        origin = np.zeros(3)
        local_pose = utils.align_vectors(mu, nr, origin, z_axis)
        ref_points = utils.transform_pc(local_points, pose=local_pose)
        return ref_points, local_points
        
    def cluster_list(self, eigen_list, bandwidth=0.00015):
        
        #bandwidth = estimate_bandwidth(eigen_list, quantile=0.1)
        print "bandwidth: ", bandwidth
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(eigen_list)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        self.num_clusters = n_clusters_
        print("number of estimated clusters : %d" % n_clusters_)
        return ms
        
        
    def create_eigen_list(self, dir_temp='temp/'):
        global_i = 0
        eigen_list = []
        origin = np.zeros(3)
        z_axis = np.array([0, 0, 1])
        for file in self.files_list:
            self.read_ply(file)
            r = self.l
            pc_samples = self.samples
            pc_sample_indices = self.sample_indices
            print "samples count: ", pc_samples.shape[0]
            for point_number, samplept in enumerate(pc_samples):
                ref_points, _ = self.get_ref_points(samplept, r)
                cov_mat_ref = utils.build_cov(ref_points, mean=origin)
                w_ref, v_ref = LA.eigh(cov_mat_ref)
                eigen_list.append(w_ref)
                #np.save(dir_temp + "sample_" + str(global_i), ref_points)
                #plotutils.show_pc(ref_points)
                #mlab.show()
                if global_i % 100 == 0:
                    print "sample: ", global_i
                global_i += 1
        np.save(dir_temp + "eigen_list", np.asarray(eigen_list))
        return np.asarray(eigen_list), global_i
        
        
    def create_dataset(self, dir_temp='temp/', num_rotations=40, bandwidth=0.000015):
        print "creating eigen list....."
        eigen_list, global_i = self.create_eigen_list(dir_temp)
        print "eigen list created" 
        z_axis = np.array([0, 0, 1])
        
        #### clustering
        print "clustering..."
        ms = self.cluster_list(eigen_list, bandwidth=0.000015)
        np.save(dir_temp + "num_clusters", self.num_clusters)
        labels = ms.labels_
        assert (global_i == len(labels))
        
        #### saving data set after clustering
        print "creating dataset"
        X = np.zeros((1*  num_rotations, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int32)
        Y = np.zeros((1*  num_rotations), np.int32)
        global_i = 0 
        for file in self.files_list:
            self.read_ply(file)
            r = self.l
            pc_samples = self.samples
            #pc_sample_indices = self.sample_indices
            print "samples count: ", pc_samples.shape[0]
            for _, samplept in enumerate(pc_samples):
                ref_points, local_points = self.get_ref_points(samplept, r)
                for aug_num, theta in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
                    if aug_num == num_rotations:
                        break
                    rot2d = utils.angle_axis_to_rotation(theta, z_axis)
                    rot_points = utils.transform_pc(ref_points, rot2d)
                    #rz = r / 2
                    rz = np.max ([np.max(ref_points[:, 2]), -np.min(ref_points[:, 2])])
                    for rot_pt in rot_points:
                                            
                        x = int(((rot_pt[0] + r) / (2 * r))*(self.patch_dim - 1))
                        y = int(((rot_pt[1] + r) / (2 * r))*(self.patch_dim - 1))
                        z = int(((rot_pt[2] + rz) / (2 * rz))*(self.patch_dim - 1))
                        
                        if (z >= 0) and (z < self.patch_dim) and (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                            X[aug_num, x, y, z, 0] = 1
                    #X[point_number*num_rotations + aug_num, :] = patch.reshape((np.power(self.patch_dim, 3),))
                    Y[aug_num] = labels[global_i]
                    np.save(dir_temp + "sample_" + str(global_i) + "_" + str(aug_num), X[aug_num])
                    np.save(dir_temp + "label_" +  str(global_i) + "_" + str(aug_num), Y[aug_num])
                if global_i % 100 == 0:
                    print "sample: ", global_i
                global_i += 1
        self.save_meta(dir_temp)
    
    def load_dataset(self, dir_temp='temp/'):
        #self.num_clusters = np.load(dir_temp + "num_clusters.npy")
        meta = np.load(dir_temp + "meta.npy")
        self.num_clusters = meta[0]
        self.num_samples = meta[1]
        self.patch_dim = meta[2]
        print [self.num_clusters, self.num_samples, self.patch_dim]
            
    
    def save_meta(self, dir_temp='temp/'):
        meta = [self.num_clusters, self.num_samples, self.patch_dim]
        np.save(dir_temp + "meta", meta)
         

    def next_batch_3d_file(self, batch_size, num_rotations=40, dir_temp='temp/'):
        X = np.zeros((batch_size*  num_rotations, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int32)
        Y = np.zeros((batch_size*  num_rotations), np.int32)
        for point_number in range(batch_size):
            for aug_num, _ in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
                if aug_num == num_rotations:
                    break
                temp_file_sample = dir_temp + "sample_" + str(self.sample_class_current) + '_' + str(aug_num) + '.npy'
                temp_file_label = dir_temp + "label_" + str(self.sample_class_current) + '_' + str(aug_num) + '.npy'
                assert(os.path.isfile(temp_file_sample))
                assert(os.path.isfile(temp_file_label))
                X[point_number*num_rotations + aug_num ] = np.load(temp_file_sample)
                Y[point_number*num_rotations + aug_num ] = np.load(temp_file_label)
            self.sample_class_current = (self.sample_class_current + 1) % self.num_samples
        return X, Y
            
        