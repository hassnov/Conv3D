import numpy as np
from numpy import linalg as LA
from plyfile import PlyData
from scipy import spatial
import os.path
from sampling import Sampler
from sampling import SampleAlgorithm
import utils
import time
import logging
#import plotutils
#from mayavi import mlab
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import enum
from scipy import ndimage


class PlyReader:
    
    data = None
    samples = None
    sample_indices = None
    num_classes = None
    index = 0
    l = None
    num_rotations = None
    patch_dim = None
    use_normals_pc = None
    use_point_as_mean = None
    flip_view_point = None
    sigma = None
    tree = None
    sample_class_start = 0
    sample_class_current = 0



    def read_ply(self, file_name, num_samples=1000, sample_class_start=0, add_noise =False,
                  noise_prob=0.3, noise_factor=0.02, sampling_algorithm=SampleAlgorithm.Uniform,
                  rotation_axis=[0, 0, 1], rotation_angle=0):
        
        #ply = PlyData.read(file_name)
        #vertex = ply['vertex']
        #(x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        #points = zip(x.ravel(), y.ravel(), z.ravel())
        #np.save('points', points)
        points = np.load('points/points_bunny.npy')
        if add_noise:
            self.data = utils.add_noise(points, prob=noise_prob, factor=noise_factor)
        else:
            self.data = np.asarray(points)
        rot = utils.angle_axis_to_rotation(rotation_angle, rotation_axis)
        self.data = utils.transform_pc(self.data, rot)
        #plotutils.show_pc(self.data)
        #mlab.show()
#TODO: better sampling
        self.samples, self.sample_indices = Sampler.sample(self.data, -1, num_samples-1, file_name="points/points_bunny.npy", pose=rot, sampling_algorithm=sampling_algorithm)
        self.tree = spatial.KDTree(self.data) 
        self.sample_class_start = sample_class_start
        self.sample_class_current = sample_class_start
        self.num_classes = self.samples.shape[0]
        logging.basicConfig(filename='example.log',level=logging.DEBUG)
        return self.data
    
    
    def set_variables(self, l, patch_dim=24, filter_bad_samples=False, filter_threshold=50, use_normals_pc=True, use_point_as_mean=False, flip_view_point=False, sigma=0.7071): 
        self.l = l
        #self.num_rotations=num_rotations
        self.patch_dim=patch_dim
        self.use_normals_pc=use_normals_pc
        self.use_point_as_mean=use_point_as_mean
        self.flip_view_point=flip_view_point
        self.sigma=sigma
        if filter_bad_samples:
            sample_indices_temp = []
            for idx in self.sample_indices:
                if self.is_good_sample(self.data[idx], filter_threshold):
                    sample_indices_temp.append(idx)
            self.sample_indices = np.asarray(sample_indices_temp)
            self.samples = self.data[self.sample_indices]
            self.num_classes = self.samples.shape[0]
        
    
    def compute_total_samples(self, num_rotations=20):
        return self.samples.shape[0]*num_rotations
    

    def is_good_sample(self, samplept, threshold=50):
        r = self.l / np.sqrt(2)
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
    
        
    def next_batch(self, batch_size, num_rotations=20, num_channels=3, num_classes=-1, d2=False):
        if d2:
            return self.next_batch_2d(batch_size, num_rotations, num_classes, num_channels)
        else:
            return self.next_batch_3d(batch_size, num_rotations, num_classes)
        
    
    def next_batch_3d(self, batch_size, num_rotations=20, num_classes=-1):
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
        
        X = np.zeros((batch_size*  num_rotations, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int32)
        Y = np.zeros((batch_size*  num_rotations), np.int32)
        
        
        r = self.l# / np.sqrt(2)
        #r = self.l / np.sqrt(2)
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
            #plotutils.show_pc(ref_points, np.zeros((1, 3)), mode='sphere', scale_factor=0.001)
            #mlab.show()
            for aug_num, theta in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
                if aug_num == num_rotations:
                    break
                rot2d = utils.angle_axis_to_rotation(theta, z_axis)
                rot_points = utils.transform_pc(ref_points, rot2d)
                #patch = np.zeros((self.patch_dim, self.patch_dim, self.patch_dim), dtype='int32')
                rz = r / 3
                #rz = r
                for rot_pt in rot_points:
                                        
                    x = int(((rot_pt[0] + r) / (2 * r))*(self.patch_dim - 1))
                    y = int(((rot_pt[1] + r) / (2 * r))*(self.patch_dim - 1))
                    z = int(((rot_pt[2] + rz) / (2 * rz))*(self.patch_dim - 1))
                    
                    #x = int(((rot_pt[0] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                    #y = int(((rot_pt[1] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                    #z = int(((rot_pt[2] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                    
                    #x = int(self.patch_dim*(rot_pt[0] / self.l) + self.patch_dim / 2)
                    #y = int(self.patch_dim*(rot_pt[1] / self.l) + self.patch_dim / 2)
                    #z = int(self.patch_dim*(rot_pt[2] / self.l) + self.patch_dim / 2)
                    if (z >= 0) and (z < self.patch_dim) and (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                        #patch[x, y, z] = 1
                        #X[point_number*num_rotations + aug_num, x + self.patch_dim * (y + self.patch_dim * z)] = 1
                        X[point_number*num_rotations + aug_num, x, y, z, 0] = 1
                #X[point_number*num_rotations + aug_num, :] = patch.reshape((np.power(self.patch_dim, 3),))
                Y[point_number*num_rotations + aug_num] = self.sample_class_current % num_classes
            #fig = plotutils.plot_patch_3D(X[point_number*num_rotations + 0], name='patch label ' + str(self.sample_class_current % num_classes))
            #plt.show()
            #TODO: start from start not 0 with sample_class_current                
            self.sample_class_current += 1
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


    def patch_3d(self, point_in):
        X = np.zeros((1, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int32)
        Y = np.zeros((1), np.int32)
        r = self.l / np.sqrt(2)
        i = self.tree.query_ball_point(point_in[0:3], r=r)
        distances, indices = self.tree.query(point_in[0:3], k=len(i))
        local_points = self.data[indices]
        if len(i) <= 8:
            return None
        if self.use_point_as_mean:
            mu = point_in[0:3]
        else:
            #TODO: compute real mean
            mu = point_in[0:3]
        if (self.use_normals_pc) and (point_in.shape == 6):
            nr = point_in[3:6]
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

        #theta = -np.pi
        #rot2d = utils.angle_axis_to_rotation(theta, z_axis)
        #rot_points = utils.transform_pc(ref_points, rot2d)
        for ref_pt in ref_points:
            x = int(((ref_pt[0] + self.l) / (2 * self.l))*(self.patch_dim - 1))
            y = int(((ref_pt[1] + self.l) / (2 * self.l))*(self.patch_dim - 1))
            z = int(((ref_pt[2] + self.l) / (2 * self.l))*(self.patch_dim - 1))
            if (z >= 0) and (z < self.patch_dim) and (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                X[0, x, y, z, 0] = 1
        Y[0] = -1
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
            #plotutils.show_pc(ref_points)
            #mlab.show()
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
                    
                    #x = int(((rot_pt[0] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                    #y = int(((rot_pt[1] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                    #z = int(((rot_pt[2] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                    
                    #x = int(self.patch_dim*(rot_pt[0] / self.l) + self.patch_dim / 2)
                    #y = int(self.patch_dim*(rot_pt[1] / self.l) + self.patch_dim / 2)
                    #z = int(self.patch_dim*(rot_pt[2] / self.l) + self.patch_dim / 2)
                    if (z >= 0) and (z < self.patch_dim) and (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                        #patch[x, y, z] = 1
                        #X[point_number*num_rotations + aug_num, x + self.patch_dim * (y + self.patch_dim * z)] = 1
                        X[point_number*num_rotations + aug_num, x, y, z, 0] = 1
                #X[point_number*num_rotations + aug_num, :] = patch.reshape((np.power(self.patch_dim, 3),))
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
    
