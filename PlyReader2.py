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
#import plotutils
#from mayavi import mlab

import enum
#from scipy.io.matlab.mio5_utils import scipy
#import scipy as sc
from scipy import ndimage


class PlyReader:
    
    data = None
    samples = None
    sample_indices = None
    num_classes = None
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
    noise_prob=0.3
    noise_factor=0.02
    #rotation of whole body
    rotation_axis=[0, 0, 1]
    rotation_angle=0
    filter_bad_samples=False
    filter_threshold=50,
    
    
    files_file = None
    files_list = []
    file_index = 0

    nr_count = 1

    
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
                  noise_prob=0.3, noise_factor=0.02, sampling_algorithm=SampleAlgorithm.Uniform, 
                  rotation_axis=[0, 0, 1], rotation_angle=0, num_classes=2000, 
                  relL=0.07, patch_dim=32, filter_bad_samples=False, filter_threshold=50,
                  nr_count=1,
                  use_normals_pc=False, use_point_as_mean=False, flip_view_point=False, sigma=0.7071):
        self.files_file = files_file
        self.add_noise = add_noise
        self.rotation_axis = rotation_axis
        self.rotation_angle = rotation_angle
        self.noise_prob = noise_prob
        self.noise_factor = noise_factor
        self.sample_class_start = sample_class_start
        self.sample_class_current = sample_class_start
        self.num_classes = num_classes
        self.sampling_algorithm = sampling_algorithm
        self.filter_bad_samples = filter_bad_samples
        self.filter_threshold = filter_threshold
        self.nr_count = nr_count
        self.fill_files_list()
        print 'files list: ', self.files_list
        
        self.relL = relL
        self.patch_dim=patch_dim
        self.use_normals_pc=use_normals_pc
        self.use_point_as_mean=use_point_as_mean
        self.flip_view_point=flip_view_point
        self.sigma=sigma
        
        
        file_name = self.files_list[self.file_index]
        print 'file: ', file_name
        self.read_ply(file_name)
        


    def read_ply(self, file_name):
        num_samples = self.num_classes // len(self.files_list)
        if self.file_index == len(self.files_list) - 1:
            num_samples = num_samples + (self.num_classes - (num_samples * len(self.files_list)))
        
        #ply = PlyData.read(file_name)
        #vertex = ply['vertex']
        #(x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        #points = zip(x.ravel(), y.ravel(), z.ravel())
        #np.save('points_vase', points)
        #points = np.load('points.npy')
        points = np.load(file_name)
        if self.add_noise:
            self.data = utils.add_noise(points, prob=self.noise_prob, factor=self.noise_factor)
        else:
            self.data = np.asarray(points)
        
        #if self.data.shape[0] > 2e5:
        #        self.data, _ = Sampler.sample(self.data, -1, 2e5, sampling_algorithm=self.sampling_algorithm)
            
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
        
        #TODO:Intergrate with num_samples for consistency
        if self.filter_bad_samples:
            temp_file_samples = 'temp/' + os.path.basename(file_name) + '_' + str(num_samples) + '_filter' + str(self.filter_threshold) + '.npy'
            print 'samples file: ', temp_file_samples 
            if os.path.isfile(temp_file_samples):
                self.sample_indices = np.load(temp_file_samples)
                self.samples = self.data[self.sample_indices]
            else:
                self.samples, self.sample_indices = Sampler.sample(self.data, -1, num_samples*2, sampling_algorithm=self.sampling_algorithm)
                self.samples = self.samples[0:num_samples*2]
                self.sample_indices = self.sample_indices[0:num_samples*2]
                sample_indices_temp = []
                for idx in self.sample_indices:
                    if self.is_good_sample(self.data[idx], self.filter_threshold):
                        sample_indices_temp.append(idx)
                        if len(sample_indices_temp) >= num_samples:
                            break   
                assert (len(sample_indices_temp) >= num_samples)
                self.sample_indices = np.asarray(sample_indices_temp[0:num_samples])
                self.samples = self.data[self.sample_indices]
                np.save(temp_file_samples, self.sample_indices)
                #plotutils.show_pc(self.samples)
                #mlab.show()
        
        logging.basicConfig(filename='example.log',level=logging.DEBUG)
        return self.data
    
    
    def compute_total_samples(self, num_rotations=20):
        return self.num_classes*num_rotations*
    
    
        
    def next_batch(self, batch_size, num_rotations=20, num_channels=3, num_classes=-1, d2=False):
        if d2:
            return self.next_batch_2d(batch_size, num_rotations, num_classes, num_channels)
        else:
            return self.next_batch_3d(batch_size, num_rotations, num_classes)
        
    
    def is_good_sample(self, samplept, threshold=50):
        r = self.l# / np.sqrt(2)
        i = self.tree.query_ball_point(samplept[0:3], r=r)
        _, indices = self.tree.query(samplept[0:3], k=len(i))
        local_points = self.data[indices]
        if self.use_point_as_mean:
            mu = samplept[0:3]
        else:
            #TODO: compute real mean
            mu = samplept[0:3]
        
        cov_mat = utils.build_cov(local_points, mean=mu)
        w, v = LA.eigh(cov_mat)
        min_id = np.argmin(w)
        isort = np.argsort(w)
        nr = np.transpose(v[:, min_id])
        nr = utils.normalize(nr)
        if w[isort[1]] / w[isort[0]] > threshold:
            return False
        return True
            
            
        
    def next_batch_3d_file(self, batch_size, num_rotations=20):
        X = np.zeros((batch_size*  num_rotations, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int32)
        Y = np.zeros((batch_size*  num_rotations), np.int32)
        for point_number in range(batch_size):
            for nr_num in range(self.nr_count):
                for aug_num, theta in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
                    if aug_num == num_rotations:
                        break
                    Y[point_number*num_rotations + aug_num] = self.sample_class_current % self.num_classes
                    temp_file = 'temp/' + str(self.sample_class_current) + '_' + str(num_rotations) + '_' + str(aug_num) + '_nr'+ str(nr_num) + '.npy'
                    assert(os.path.isfile(temp_file))
                    X[point_number*num_rotations + aug_num] = np.load(temp_file)
                    #print 'file loaded: ', temp_file
                    
            self.sample_class_current = (self.sample_class_current + 1) % self.num_classes
        return X, Y
            
        
        
    def next_batch_3d(self, batch_size, num_rotations=20):
        logging.info('index: ' + str(self.index) + '    current_label: ' + str(self.sample_class_current % self.num_classes) )
        if self.index + batch_size <= self.samples.shape[0]:
            pc_samples = self.samples[self.index:self.index+batch_size]
            self.index += batch_size
        else:
            pc_samples = self.samples[self.index:self.samples.shape[0]]
            self.index = self.index + batch_size -self.samples.shape[0]
            self.next_file()
            pc_samples = np.vstack((pc_samples, self.samples[0:self.index]))
        
        X = np.zeros((batch_size*  num_rotations, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int32)
        Y = np.zeros((batch_size*  num_rotations), np.int32)
        
        #r = self.l / np.sqrt(2)
        r = self.l# / np.sqrt(2)
        for point_number, samplept in enumerate(pc_samples):
            
            i = self.tree.query_ball_point(samplept[0:3], r=r)
            _, indices = self.tree.query(samplept[0:3], k=len(i))
            #query_time = time.time() - start_time
            #print "query time: {0:.2f}".format(query_time)
            local_points = self.data[indices]
            #if local_points.shape[0] > 1000:
            #    local_points, _ = Sampler.sample(local_points, -1, 1000, sampling_algorithm=self.sampling_algorithm)
            if len(i) <= 8:
                continue
            if self.use_point_as_mean:
                mu = samplept[0:3]
            else:
                #TODO: compute real mean
                mu = np.zeros(3)
                point_count = local_points.shape[0]
                mu[0] = np.sum(local_points[:, 0]) / point_count
                mu[1] = np.sum(local_points[:, 1]) / point_count
                mu[2] = np.sum(local_points[:, 2]) / point_count
            if (self.use_normals_pc) and (samplept.shape == 6):
                nr = samplept[3:6]
                #TODO: get_weighted_normal
            else:# calc normals
                cov_mat = utils.build_cov(local_points, mean=mu)
                w, v = LA.eigh(cov_mat)
                min_id = np.argmin(w)
                #print 'eigenvalues: ', w
                nr1 = np.transpose(v[:, min_id])
                nr1 = utils.normalize(nr1)
                if self.nr_count > 1:
                    nr2 = [-x for x in nr1]
                    nrs = [nr1, nr2]
                else:
                    nrs = [nr1]
            
    # at this point the plane is defined by (mu, nr)
    # we transform the mean to be (0,0,0) and the normal to be  (0,0,1)
    # to obtain canonical frame
            z_axis = np.array([0, 0, 1])
            origin = np.zeros(3)
            for nr_num, nr in enumerate(nrs):
                local_pose = utils.align_vectors(mu, nr, origin, z_axis)
                ref_points = utils.transform_pc(local_points, pose=local_pose)
                #plotutils.show_pc(ref_points, np.zeros((1, 3)), mode='sphere', scale_factor=0.001)
                #mlab.show()
                start_time = time.time()
                for aug_num, theta in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
                    if aug_num == num_rotations:
                        break
                    rot2d = utils.angle_axis_to_rotation(theta, z_axis)
                    rot_points = utils.transform_pc(ref_points, rot2d)
                    #patch = np.zeros((self.patch_dim, self.patch_dim, self.patch_dim), dtype='int32')
                    #TODO: use numpy.apply_along_axis
                    rz = r / 3
                    Y[point_number*num_rotations + aug_num] = self.sample_class_current % self.num_classes
                    temp_file = 'temp/' + str(self.sample_class_current) + '_' + str(num_rotations) + '_' + str(aug_num) + '_nr'+ str(nr_num) +'.npy'
                    if os.path.isfile(temp_file):
                        X[point_number*num_rotations + aug_num] = np.load(temp_file)
                        #print 'file loaded: ', temp_file
                        continue
                    
                    for rot_pt in rot_points:
                        x = int(((rot_pt[0] + r) / (2 * r))*(self.patch_dim - 1))
                        y = int(((rot_pt[1] + r) / (2 * r))*(self.patch_dim - 1))
                        z = int(((rot_pt[2] + rz) / (2 * rz))*(self.patch_dim - 1))
                        if (z >= 0) and (z < self.patch_dim) and (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                            X[point_number*num_rotations + aug_num, x, y, z, 0] = 1
                        
                    #X[point_number*num_rotations + aug_num, :, :, :, 0] = ndimage.morphology.distance_transform_edt(1 - X[point_number*num_rotations + aug_num, :, :, :, 0])
                    #X[point_number*num_rotations + aug_num, :, :, :, 0] /= np.max(X[point_number*num_rotations + aug_num, :, :, :, 0])
                            
                    np.save(temp_file, X[point_number*num_rotations + aug_num])
                    
                #fig = plotutils.plot_patch_3D(X[point_number*num_rotations + 0], name='patch label ' + str(self.sample_class_current % num_classes))
                #plt.show()
                #TODO: start from start not 0 with sample_class_current                
            self.sample_class_current = (self.sample_class_current + 1) % self.num_classes
        return X, Y

 
    def next_batch_2d(self, batch_size, num_rotations=20, num_classes=-1, num_channels=3):
        if num_classes == -1:
            num_classes = self.num_classes
        if self.index + batch_size < self.samples.shape[0]:
            pc_samples = self.samples[self.index:self.index+batch_size]
            self.index += batch_size
        else:
            pc_samples = self.samples[self.index:self.samples.shape[0]]
            self.index = self.index + batch_size -self.samples.shape[0]
            #self.sample_class_current = self.sample_class_start
            pc_samples = np.vstack((pc_samples, self.samples[0:self.index]))
            
        X = np.zeros((batch_size*  num_rotations, self.patch_dim, self.patch_dim, num_channels))
        Y = np.zeros((batch_size*  num_rotations))
        
        r = self.l / np.sqrt(2)
        
        for point_number, samplept in enumerate(pc_samples):
            i = self.tree.query_ball_point(samplept[0:3], r=r)
            distances, indices = self.tree.query(samplept[0:3], k=len(i))
            local_points = self.data[indices]
            if len(i) <= 8:
                continue
            if self.use_point_as_mean:
                mu = samplept[0:3]
            else:
                #TODO: compute real mean
                mu = samplept[0:3]
            if (self.use_normals_pc) and (samplept.shape == 6):
                nr = samplept[3:6]
                #TODO: get_weighted_normal
            else:# calc normals
                cov_mat = utils.build_cov(local_points, mean=mu)
                w, v = LA.eigh(cov_mat)
                min_id = np.argmin(w)
                nr = np.transpose(v[:, min_id])
                nr = utils.normalize(nr)
                #nr = [-x for x in nr]
            
    # at this point the plane is defined by (mu, nr)
    # we transform the mean to be (0,0,0) and the normal to be  (0,0,1)
    # to obtain canonical frame
            z_axis = np.array([0, 0, 1])
            origin = np.zeros(3)
            local_pose = utils.align_vectors(mu, nr, origin, z_axis)
            ref_points = utils.transform_pc(local_points, pose=local_pose)

            for aug_num, theta in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
                if aug_num == num_rotations:
                    break
                rot2d = utils.angle_axis_to_rotation(theta, z_axis)
                rot_points = utils.transform_pc(ref_points, rot2d)
                #plotutils.show_pc(rot_points)
                #plotutils.draw_normal(origin, z_axis)
                #mlab.show()
                #patch = np.zeros((self.patch_dim, self.patch_dim, self.patch_dim), dtype='int32')
                if num_channels == 3:
                    for rot_pt in rot_points:
                        #project on z
                        x = int(((rot_pt[0] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                        y = int(((rot_pt[1] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                        #x = int(self.patch_dim*(rot_pt[0] / self.l) + self.patch_dim / 2)
                        #y = int(self.patch_dim*(rot_pt[1] / self.l) + self.patch_dim / 2)
                        if (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                            X[point_number*num_rotations + aug_num, x, y, 2] = 1 # rot_pt[2]
                        
                        #project on y
                        #x = int(self.patch_dim*(rot_pt[0] / self.l) + self.patch_dim / 2)
                        #z = int(self.patch_dim*(rot_pt[2] / self.l) + self.patch_dim / 2)
                        x = int(((rot_pt[0] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                        z = int(((rot_pt[2] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                        if (z >= 0) and (z < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                            X[point_number*num_rotations + aug_num, z, x, 1] = 1 #rot_pt[1]
                        
                        #project on x
                        #y = int(self.patch_dim*(rot_pt[1] / self.l) + self.patch_dim / 2)
                        #z = int(self.patch_dim*(rot_pt[2] / self.l) + self.patch_dim / 2)
                        y = int(((rot_pt[1] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                        z = int(((rot_pt[2] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                        if (z >= 0) and (z < self.patch_dim) and (y >= 0) and (y < self.patch_dim):
                            X[point_number*num_rotations + aug_num, z, y, 0] = 1 #rot_pt[0]
                else:
                    for rot_pt in rot_points:
                        #project on z
                        #x = int(self.patch_dim*(rot_pt[0] / self.l) + self.patch_dim / 2)
                        #y = int(self.patch_dim*(rot_pt[1] / self.l) + self.patch_dim / 2)
                        x = int(((rot_pt[0] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                        y = int(((rot_pt[1] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                        if (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                            X[point_number*num_rotations + aug_num, x, y, 0] = rot_pt[2]
                    #plt.imshow(X[point_number*num_rotations + aug_num].reshape([self.patch_dim, self.patch_dim,]), cmap = 'gray', interpolation = 'bicubic')
                    #plt.title('label: ' + str(self.sample_class_current % num_classes))
                    #plt.show()
                Y[point_number*num_rotations + aug_num] = self.sample_class_current % num_classes
                #patches.append(patch)
            self.sample_class_current += 1
        return X, Y


    def next_batch_3d_sdf(self, batch_size, num_rotations=20, num_classes=-1):
        if num_classes == -1:
            num_classes = self.num_classes
        logging.info('index: ' + str(self.index) + '    current_label: ' + str(self.sample_class_current % num_classes) )
        if self.index + batch_size < self.samples.shape[0]:
            pc_samples = self.samples[self.index:self.index+batch_size]
            self.index += batch_size
        else:
            pc_samples = self.samples[self.index:self.samples.shape[0]]
            self.index = self.index + batch_size -self.samples.shape[0]
            #self.sample_class_current = self.sample_class_start
            pc_samples = np.vstack((pc_samples, self.samples[0:self.index]))
        
        X = np.zeros((batch_size*  num_rotations, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.float32)
        Y = np.zeros((batch_size*  num_rotations), np.int32)
        
        r = self.l #/ np.sqrt(2)
        rz = r / 3
        for point_number, samplept in enumerate(pc_samples):
            i = self.tree.query_ball_point(samplept[0:3], r=r)
            distances, indices = self.tree.query(samplept[0:3], k=len(i))
            local_points = self.data[indices]
            if len(i) <= 8:
                continue
            if self.use_point_as_mean:
                mu = samplept[0:3]
            else:
                #TODO: compute real mean
                mu = samplept[0:3]
                point_count = local_points.shape[0]
                mu[0] = np.sum(local_points[:, 0]) / point_count
                mu[1] = np.sum(local_points[:, 1]) / point_count
                mu[2] = np.sum(local_points[:, 2]) / point_count
            if (self.use_normals_pc) and (samplept.shape == 6):
                nr = samplept[3:6]
                #TODO: get_weighted_normal
            else:# calc normals
                cov_mat = utils.build_cov(local_points, mean=mu)
                w, v = LA.eigh(cov_mat)
                min_id = np.argmin(w)
                #print 'eigenvalues: ', w
                nr = np.transpose(v[:, min_id])
                nr = utils.normalize(nr)
                #nr = [-x for x in nr]
            
    # at this point the plane is defined by (mu, nr)
    # we transform the mean to be (0,0,0) and the normal to be  (0,0,1)
    # to obtain canonical frame
            z_axis = np.array([0, 0, 1])
            origin = np.zeros(3)
            local_pose = utils.align_vectors(mu, nr, origin, z_axis)
            ref_points = utils.transform_pc(local_points, pose=local_pose)
            plotutils.show_pc(self.data, local_points)
            mlab.show()
            for aug_num, theta in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
                if aug_num == num_rotations:
                    break
                rot2d = utils.angle_axis_to_rotation(theta, z_axis)
                rot_points = utils.transform_pc(ref_points, rot2d)
                #patch = np.zeros((self.patch_dim, self.patch_dim, self.patch_dim), dtype='int32')
                for rot_pt in rot_points:
                    x = int(((rot_pt[0] + r) / (2 * r))*(self.patch_dim - 1))
                    y = int(((rot_pt[1] + r) / (2 * r))*(self.patch_dim - 1))
                    z = int(((rot_pt[2] + rz) / (2 * rz))*(self.patch_dim - 1))
                    if (z >= 0) and (z < self.patch_dim) and (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                        X[point_number*num_rotations + aug_num, x, y, z, 0] = 1
                Y[point_number*num_rotations + aug_num] = self.sample_class_current % num_classes
                
                X[point_number*num_rotations + aug_num, :, :, :, 0] = ndimage.morphology.distance_transform_edt(1 - X[point_number*num_rotations + aug_num, :, :, :, 0])
                X[point_number*num_rotations + aug_num, :, :, :, 0] /= np.max(X[point_number*num_rotations + aug_num, :, :, :, 0])
            #fig = plotutils.plot_patch_TSDF(X[point_number*num_rotations, :, :, :, 0], name='patch label ' + str(self.sample_class_current % num_classes))
            #fig = plotutils.plot_patch_3D(X[point_number*num_rotations + 0], name='patch label ' + str(self.sample_class_current % num_classes))
            #plt.show()
            #mlab.show()
            #TODO: start from start not 0 with sample_class_current                
            self.sample_class_current += 1
        return X, Y

        
        
