import numpy as np
from numpy import linalg as LA
from plyfile import PlyData
from scipy import spatial
#import os.path
from sampling import Sampler
import utils


class PlyReader:
    
    data = None
    samples = None
    num_classes = 1
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


    def read_ply(self, file_name, num_samples=1000, sample_class_start=0):
        ply = PlyData.read(file_name)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        points = zip(x.ravel(), y.ravel(), z.ravel())
        self.data = np.asarray(points)
#TODO: better sampling
        self.samples = Sampler.sample(self.data, -1, num_samples)
        self.tree = spatial.KDTree(self.data) 
        self.sample_class_start = sample_class_start
        self.sample_class_current = sample_class_start
        self.num_classes = self.samples.shape[0]
        return self.data
    
    
    def set_variables(self, l, num_rotations=20, patch_dim=24, use_normals_pc=True, use_point_as_mean=False, flip_view_point=False, sigma=0.7071): 
        self.l = l
        self.num_rotations=num_rotations
        self.patch_dim=patch_dim
        self.use_normals_pc=use_normals_pc
        self.use_point_as_mean=use_point_as_mean
        self.flip_view_point=flip_view_point
        self.sigma=sigma
        
    
    def compute_total_samples(self, num_rotations=20):
        return self.samples.shape[0]*num_rotations
    
    
        
    def next_batch(self, batch_size, num_rotations=20, num_channels=3, num_classes=-1, d2=False):
        if d2:
            return self.next_batch_2d(batch_size, num_rotations, num_classes, num_channels)
        else:
            return self.next_batch_3d(batch_size, num_rotations, num_classes)
        
    
    def next_batch_3d(self, batch_size, num_rotations=20, num_classes=-1):
        if num_classes == -1:
            num_classes = self.num_classes
        if self.index + batch_size < self.samples.shape[0]:
            pc_samples = self.samples[self.index:self.index+batch_size]
            self.index += batch_size
        else:
            pc_samples = self.samples[self.index:self.samples.shape[0]]
            self.index = self.index + batch_size -self.samples.shape[0]
            self.sample_class_current = self.sample_class_start
            pc_samples = np.vstack((pc_samples, self.samples[0:self.index]))
            
        X = np.zeros((batch_size*  num_rotations, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int32)
        Y = np.zeros((batch_size*  num_rotations), np.int32)
        
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

            for aug_num, theta in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/self.num_rotations)):
                if aug_num == num_rotations:
                    break
                rot2d = utils.angle_axis_to_rotation(theta, z_axis)
                rot_points = utils.transform_pc(ref_points, rot2d)
                #patch = np.zeros((self.patch_dim, self.patch_dim, self.patch_dim), dtype='int32')
                for rot_pt in rot_points:
                    x = int(((rot_pt[0] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                    y = int(((rot_pt[1] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                    z = int(((rot_pt[2] + self.l) / (2 * self.l))*(self.patch_dim - 1))
                    #x = int(self.patch_dim*(rot_pt[0] / self.l) + self.patch_dim / 2)
                    #y = int(self.patch_dim*(rot_pt[1] / self.l) + self.patch_dim / 2)
                    #z = int(self.patch_dim*(rot_pt[2] / self.l) + self.patch_dim / 2)
                    if (z >= 0) and (z < self.patch_dim) and (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                        #patch[x, y, z] = 1
                        #X[point_number*num_rotations + aug_num, x + self.patch_dim * (y + self.patch_dim * z)] = 1
                        X[point_number*num_rotations + aug_num, x, y, z, 0] = 1
                #X[point_number*num_rotations + aug_num, :] = patch.reshape((np.power(self.patch_dim, 3),))
                Y[point_number*num_rotations + aug_num] = self.sample_class_current % num_classes
                #patches.append(patch)
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

            for aug_num, theta in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/self.num_rotations)):
                if aug_num == num_rotations:
                    break
                rot2d = utils.angle_axis_to_rotation(theta, z_axis)
                rot_points = utils.transform_pc(ref_points, rot2d)
                #patch = np.zeros((self.patch_dim, self.patch_dim, self.patch_dim), dtype='int32')
                if num_channels == 3:
                    for rot_pt in rot_points:
                        #project on z
                        x = int(self.patch_dim*(rot_pt[0] / self.l) + self.patch_dim / 2)
                        y = int(self.patch_dim*(rot_pt[1] / self.l) + self.patch_dim / 2)
                        if (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                            X[point_number*num_rotations + aug_num, x, y, 2] = 1 # rot_pt[2]
                        
                        #project on y
                        x = int(self.patch_dim*(rot_pt[0] / self.l) + self.patch_dim / 2)
                        z = int(self.patch_dim*(rot_pt[2] / self.l) + self.patch_dim / 2)
                        if (z >= 0) and (z < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                            X[point_number*num_rotations + aug_num, z, x, 1] = 1 #rot_pt[1]
                        
                        #project on x
                        y = int(self.patch_dim*(rot_pt[1] / self.l) + self.patch_dim / 2)
                        z = int(self.patch_dim*(rot_pt[2] / self.l) + self.patch_dim / 2)
                        if (z >= 0) and (z < self.patch_dim) and (y >= 0) and (y < self.patch_dim):
                            X[point_number*num_rotations + aug_num, z, y, 0] = 1 #rot_pt[0]
                else:
                    for rot_pt in rot_points:
                        #project on z
                        x = int(self.patch_dim*(rot_pt[0] / self.l) + self.patch_dim / 2)
                        y = int(self.patch_dim*(rot_pt[1] / self.l) + self.patch_dim / 2)
                        if (y >= 0) and (y < self.patch_dim) and (x >= 0) and (x < self.patch_dim):
                            X[point_number*num_rotations + aug_num, x, y, 0] = 1 # rot_pt[2]
                    
                Y[point_number*num_rotations + aug_num] = self.sample_class_current % num_classes
                #patches.append(patch)
            self.sample_class_current += 1
        return X, Y
    

    
