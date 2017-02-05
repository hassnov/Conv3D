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
import sys
#import plotutils
#from mayavi import mlab
import enum

from joblib import Parallel, delayed
import multiprocessing
from scipy import ndimage

from sklearn.cluster import KMeans




class ClusterPoints:
    
    data = None
    normals = None
    samples = None
    sample_indices = None
    num_samples = None
    num_clusters = None
    labels = None
    index = 0
    tree = None
    #fpfh = None
    
    l = None
    relL=None   
    pc_diameter= None 
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
    
    samples_permutation = None
    num_permutations = None
    permutation_num = 0
    #nr_count = 1
    
    
    samples_permutation_train = None
    num_permutations_train = None
    permutation_num_train = 0
    
    samples_permutation_test = None
    num_permutations_test = None
    permutation_num_test = 0

    
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
        
        #load normals
        if os.path.isfile(root + "_normals" + ".ply"):
            if not os.path.isfile(root + "_normals" + ".npy"):
                ply1 = PlyData.read(root + "_normals" + ".ply")
                vertex = ply1['vertex']
                (nx, ny, nz) = (vertex[t] for t in ('nx', 'ny', 'nz'))
                self.normals = np.asarray(zip(nx.ravel(), ny.ravel(), nz.ravel()))
                np.save(root + "_normals" + ".npy", self.normals)
            else:
                self.normals = np.load(root + "_normals" + ".npy")
        
        if self.add_noise:
            self.data = utils.add_noise_normal(points, std=self.nois_std)
        else:
            self.data = np.asarray(points)
        
        self.pc_diameter = utils.get_pc_diameter(self.data)
        self.l = self.relL*self.pc_diameter
        
        rot = utils.angle_axis_to_rotation(self.rotation_angle, self.rotation_axis)
        self.data = utils.transform_pc(self.data, rot)
        
        #plotutils.show_pc(self.data)
        #mlab.show()
                
        #TODO: better sampling
        print "sampling file: ", file_name
        self.samples, self.sample_indices = Sampler.sample(self.data, -1, min_num_point=-1, file_name=file_name, sampling_algorithm=self.sampling_algorithm)
        #self.samples, self.sample_indices = Sampler.sample(self.data, -1, num_samples, file_name=file_name, sampling_algorithm=self.sampling_algorithm)
        #self.samples = self.samples[0:num_samples]
        #self.sample_indices = self.sample_indices[0:num_samples]
        
        self.tree = spatial.KDTree(self.data)
        return self.data
    
    
    def compute_total_samples(self, num_aug):
        return self.num_samples*num_aug
        
    
    def get_ref_points(self, samplept, r, noise_level, model_resolution, flip_nr=False):
        
        i = self.tree.query_ball_point(samplept[0:3], r=r)
        distances, indices = self.tree.query(samplept[0:3], k=len(i))
        
        #query_time = time.time() - start_time
        #print "query time: {0:.2f}".format(query_time)
        local_points1 = self.data[indices]
        
        #if local_points1.shape[0] > 5000:
        #    local_points1, _ = Sampler.sample(local_points1, -1, 5000, sampling_algorithm=SampleAlgorithm.Uniform)
        
        
        if noise_level > 0:
            local_points = utils.add_noise_normal(local_points1, mr=model_resolution, std=noise_level)
        else:
            local_points = local_points1
        #print "local points shape: ", local_points.shape
        
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
        if (self.use_normals_pc): #and type(self.normals).__module__ == np.__name__: #(self.normals != None):
            print "meshlab normal"
            min_d = np.argmin(distances)
            #if distances[min_d] > 0:
            #    raise
            nr = self.normals[indices[min_d]] 
            #nr = samplept[3:6]
            #TODO: get_weighted_normal
        else:# calc normals
            cov_mat = utils.build_cov(local_points, mean=mu)
            w, v = LA.eigh(cov_mat)
            min_id = np.argmin(w)
            #print 'eigenvalues: ', w
            nr1 = np.transpose(v[:, min_id])
            nr = utils.normalize(nr1)
            if flip_nr:
                nr = [-x for x in nr]
            
        z_axis = np.array([0, 0, 1])
        origin = np.zeros(3)
        local_pose = utils.align_vectors(mu, nr, origin, z_axis)
        ref_points = utils.transform_pc(local_points, pose=local_pose)
        return ref_points, local_points
    
    def process_all_rotations(self,dir_temp, X, global_i, ref_points, num_rotations, r, noise_i, relR_i, nr_i, augs):
        z_axis = np.array([0, 0, 1])
        thetas = np.asarray(np.arange(-np.pi, np.pi, (2*np.pi)/num_rotations))
        thetas = thetas[0:num_rotations]
        num_cores = multiprocessing.cpu_count()
        #print "num_cores: ", num_cores
        Parallel(n_jobs=num_cores)(delayed(process_rotation)(dir_temp, X, rot_i, global_i, ref_points, num_rotations, r, noise_i, relR_i, nr_i, augs,thetas, z_axis, self.patch_dim) for rot_i in range(num_rotations))
        #print Parallel(n_jobs=num_cores)(delayed(np.sqrt)(i) for i in range(10))
        
    def cluster_list_kmeans(self, eigen_list, k):
        km = KMeans(k).fit(eigen_list)
        self.labels = km.labels_
        labels_unique = np.unique(self.labels)
        n_clusters_ = len(labels_unique)
        self.num_clusters = n_clusters_
        print "kmeans clusters: ", self.num_clusters 
        return self.labels, self.num_clusters
        #bandwidth = estimate_bandwidth(eigen_list, quantile=0.5)
        
        
    def create_dataset(self, dir_temp='temp/', num_rotations=40, noises=[0, 0.3, 0.5], relRs=[0.05, 0.07], nrs=1):
        z_axis = np.array([0, 0, 1])
        #### saving data set after clustering
        print "creating dataset..."
        X = np.zeros((num_rotations * len(noises) * len(relRs)*nrs, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int8)
        #Y = np.zeros((1*  num_rotations), np.int32)
        #global_i = 4116
        global_i = 0
        #13205 1162
        #13614 1163
        for file1 in self.files_list:
            self.read_ply(file1)
            t1 = time.time()
            mr = utils.model_resolution(self.data)
            print "model resolution time: ", (time.time() - t1), "    model resolution: ", mr
            #r = self.l
            pc_samples = self.samples
            augs = np.arange(num_rotations * len(noises) * len(relRs) * nrs).reshape((len(noises), len(relRs), num_rotations, nrs))
            
            print "samples count: ", pc_samples.shape[0]
            for _, samplept in enumerate(pc_samples):
                t0 = time.time()
                X.fill(0)
                if global_i < 0:
                    print "ignore sample: ", global_i
                    global_i += 1
                    continue
                for noise_i, noise_level in enumerate(noises):
                    for relR_i, relR in enumerate(relRs):
                        for nr_i in range(nrs):
                            r = self.pc_diameter*relR
                            #Y.fill(0)
                            ref_points, _ = self.get_ref_points(samplept, r, noise_level=noise_level, model_resolution=mr, flip_nr=nr_i%2!=0)
                            
                            self.process_all_rotations(dir_temp, X, global_i, ref_points, num_rotations, r, noise_i, relR_i, nr_i, augs)
                        
                print "sample: ", global_i, " in {0:.2f} seconds".format(time.time() - t0)
                sys.stdout.flush()
                global_i += 1
        #self.save_meta(global_i, dir_temp)
    
    
    def load_dataset_and_labels(self, k, dir_temp='temp/', create_labels=False, meta_name="meta.npy", suffix=""):
        cluster_file = dir_temp + "labels_train_ae" + str(k) + ".npy"
        if os.path.isfile(cluster_file):
            print "loading saved clusters"
            self.labels = np.load(cluster_file)
            labels_unique = np.unique(self.labels)
            n_clusters_ = len(labels_unique)
            self.num_clusters = n_clusters_
        else:
            if create_labels:
                print "clustering"
                list = np.load(dir_temp + 'fpfh_list.npy')
                #list = np.load(dir_temp + 'fpfh_list.npy')
                self.cluster_list_kmeans(list, k=k)
                np.save(cluster_file, self.labels)
            else:
                print "cant find clusters"            
        
        print "kmeans clusters: ", self.num_clusters
        
        meta = np.load(dir_temp + meta_name)
        print "meta: ", meta
        #self.num_clusters = meta[0]
        self.num_samples = meta[1]
        #self.num_samples = 3440
        #self.num_samples = 3100 
        #self.num_samples = 2000
        #self.num_samples = 1400
        self.patch_dim = meta[2]
        
        #self.num_clusters = self.num_samples
        #self.labels = range(self.num_samples)
        if(self.labels == None):
            self.labels = range(self.num_samples)
            self.num_clusters = self.num_samples
        print "num_clusters: ", self.num_clusters
        #if(self.num_clusters == None):
        #    self.num_clusters = self.num_samples
            
        permutation_file = dir_temp + "permutation"+ suffix + ".npy"
        if not os.path.isfile(permutation_file):
            #num_augs = 180*2
            num_augs = meta[3]
            print "creating permutaions for {0} samples and {1} augs".format(self.num_samples, num_augs)
            self.num_permutations = self.num_samples * num_augs
            self.samples_permutation = np.random.permutation(self.num_permutations)
            np.save(permutation_file, self.samples_permutation)
            
        else:
            self.samples_permutation = np.load(permutation_file)
            self.num_permutations = self.samples_permutation.shape[0]
            num_augs = self.num_permutations / self.num_samples
            print "................num augs: ", num_augs 
            #self.samples_permutation_train = np.load(dir_temp + "permutation_train.npy")
            #self.num_permutations_train = self.samples_permutation_train.shape[0]
            #self.samples_permutation_test = np.load(dir_temp + "permutation_test.npy")
            #self.num_permutations_test = self.samples_permutation_test.shape[0]
            
        self.num_permutations_train = int(0.90*self.num_permutations)
        self.samples_permutation_train = self.samples_permutation[0:self.num_permutations_train]
        self.num_permutations_test = self.num_permutations - self.num_permutations_train
        self.samples_permutation_test = self.samples_permutation[self.num_permutations_train:]
        assert (self.num_permutations_test == self.samples_permutation_test.shape[0])
        #print "test permutations: ",  self.samples_permutation_test
        #np.save(dir_temp + "permutation_train.npy", self.samples_permutation_train)
        #np.save(dir_temp + "permutation_test.npy", self.samples_permutation_test)
        
        #logging.basicConfig(filename='example.log',level=logging.DEBUG)
        #print self.labels 
        print [self.num_clusters, self.num_samples, self.patch_dim]
        
    
    def save_meta(self, global_i, dir_temp='temp/'):
        meta = [self.num_clusters, global_i, self.patch_dim]
        np.save(dir_temp + "meta", meta)
         

    def next_batch_3d_file(self, batch_size, num_rotations=40, dir_temp='temp/'):
        X = np.zeros((batch_size*  num_rotations, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int8)
        Y = np.zeros((batch_size*  num_rotations), np.int16)
        
        for point_number in range(batch_size):
            #for aug_num, _ in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
            sample_id = self.sample_class_start + self.sample_class_current
            for aug_num in range(num_rotations):
                if aug_num == num_rotations:
                    break
                temp_file_sample = dir_temp + "sample_" + str(sample_id) + '_' + str(aug_num) + '.npy'
                #temp_file_label = dir_temp + "label_" + str(self.sample_class_current) + '_' + str(aug_num) + '.npy'
                assert(os.path.isfile(temp_file_sample))
                
                X[point_number*num_rotations + aug_num ] = np.load(temp_file_sample)
                #Y[point_number*num_rotations + aug_num ] = np.load(temp_file_label)
                Y[point_number*num_rotations + aug_num ] = self.labels[sample_id]
                #logging.info('sample: ' + str(sample_id) + '    label: ' + str(self.labels[sample_id]) + '    file: ' + temp_file_sample)
            self.sample_class_current = (self.sample_class_current + 1) % self.num_samples
        return X, Y
    
    
    def next_batch_3d_file_random(self, batch_size_full, num_augs=40, dir_temp='temp/'):
        X = np.zeros((batch_size_full, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int8)
        Y = np.zeros((batch_size_full), np.int16)
        augs = []
        ids = []
        for point_number in range(batch_size_full):
            #for aug_num, _ in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
            abs_id = self.samples_permutation[self.permutation_num]
            sample_id = abs_id // num_augs
            aug_num = abs_id % num_augs
            augs.append(aug_num)
            ids.append(sample_id)
            #sample_id = self.sample_class_start + self.sample_class_current
            temp_file_sample = dir_temp + "sample_" + str(sample_id) + '_' + str(aug_num) + '.npy'
            #temp_file_label = dir_temp + "label_" + str(self.sample_class_current) + '_' + str(aug_num) + '.npy'
            assert(os.path.isfile(temp_file_sample))
            
            X[point_number] = np.load(temp_file_sample)
            #Y[point_number*num_rotations + aug_num ] = np.load(temp_file_label)
            Y[point_number] = self.labels[sample_id]
            #logging.info('sample: ' + str(sample_id) + '    label: ' + str(self.labels[sample_id]) + '    file: ' + temp_file_sample)
            #self.sample_class_current = (self.sample_class_current + 1) % self.num_samples
            self.permutation_num = (self.permutation_num + 1) % self.num_permutations
        return X, Y, augs, ids
    
    def next_batch_3d_file_random_test(self, batch_size_full, num_augs=40, dir_temp='temp/'):
        X = np.zeros((batch_size_full, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int8)
        Y = np.zeros((batch_size_full), np.int16)
        augs = []
        ids = []
        for point_number in range(batch_size_full):
            #for aug_num, _ in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
            abs_id = self.samples_permutation_test[self.permutation_num_test]
            sample_id = abs_id // num_augs
            aug_num = abs_id % num_augs
            augs.append(aug_num)
            ids.append(sample_id)
            #sample_id = self.sample_class_start + self.sample_class_current
            temp_file_sample = dir_temp + "sample_" + str(sample_id) + '_' + str(aug_num) + '.npy'
            #temp_file_label = dir_temp + "label_" + str(self.sample_class_current) + '_' + str(aug_num) + '.npy'
            assert(os.path.isfile(temp_file_sample))
            
            X[point_number] = np.load(temp_file_sample)
            #Y[point_number*num_rotations + aug_num ] = np.load(temp_file_label)
            Y[point_number] = self.labels[sample_id]
            #logging.info('sample: ' + str(sample_id) + '    label: ' + str(self.labels[sample_id]) + '    file: ' + temp_file_sample)
            #self.sample_class_current = (self.sample_class_current + 1) % self.num_samples
            self.permutation_num_test = (self.permutation_num_test + 1) % self.num_permutations_test
        return X, Y, augs, ids
    
    def     next_batch_3d_file_random_train(self, batch_size_full, num_augs=40, dir_temp='temp/'):
        X = np.zeros((batch_size_full, self.patch_dim, self.patch_dim, self.patch_dim, 1), np.int8)
        Y = np.zeros((batch_size_full), np.int16)
        augs = []
        ids = []
        for point_number in range(batch_size_full):
            #for aug_num, _ in enumerate(utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
            abs_id = self.samples_permutation_train[self.permutation_num_train]
            sample_id = abs_id // num_augs
            aug_num = abs_id % num_augs
            augs.append(aug_num)
            ids.append(sample_id)
            #sample_id = self.sample_class_start + self.sample_class_current
            temp_file_sample = dir_temp + "sample_" + str(sample_id) + '_' + str(aug_num) + '.npy'
            #temp_file_label = dir_temp + "label_" + str(self.sample_class_current) + '_' + str(aug_num) + '.npy'
            assert(os.path.isfile(temp_file_sample))
            
            X[point_number] = np.load(temp_file_sample)
            #Y[point_number*num_rotations + aug_num ] = np.load(temp_file_label)
            Y[point_number] = self.labels[sample_id]
            #logging.info('sample: ' + str(sample_id) + '    label: ' + str(self.labels[sample_id]) + '    file: ' + temp_file_sample)
            #self.sample_class_current = (self.sample_class_current + 1) % self.num_samples
            self.permutation_num_train = (self.permutation_num_train + 1) % self.num_permutations_train
        return X, Y, augs, ids

def process_rotation(dir_temp, X, rot_i, global_i, ref_points, num_rotations, r, noise_i, relR_i, nr_i, augs, thetas, z_axis, patch_dim):
    aug_num = augs[noise_i, relR_i, rot_i, nr_i]
    theta = thetas[rot_i]
    rot2d = utils.angle_axis_to_rotation(theta, z_axis)
    rot_points = utils.transform_pc(ref_points, rot2d)
    use_tsdf = False
    #rz = np.max ([np.max(ref_points[:, 2]), -np.min(ref_points[:, 2])])
    rz = r
    xs1 = np.asarray(((rot_points[:, 0] + r) / (2 * r))*(patch_dim - 1), np.int16)
    ys1 = np.asarray(((rot_points[:, 1] + r) / (2 * r))*(patch_dim - 1), np.int16)
    zs1 = np.asarray(((rot_points[:, 2] + rz) / (2 * rz))*(patch_dim - 1), np.int16)
    above32 = np.logical_and (np.logical_and(xs1<patch_dim, ys1<patch_dim), zs1<patch_dim) 
    xs = xs1[above32]
    ys = ys1[above32]
    zs = zs1[above32]
    X1 = np.zeros((patch_dim, patch_dim, patch_dim, 1), np.int8)
    X1[xs, ys, zs, 0] = 1
    #np.save(dir_temp + "sample_" + str(global_i) + "_" + str(aug_num), X1)
    
    """
    return None
    try:
        X1[xs, ys, zs, 0] = 1
        #np.save(dir_temp + "sample_" + str(global_i) + "_" + str(aug_num), X1)
        #print "file saved: ", "sample_" + str(global_i) + "_" + str(aug_num)
    except IndexError as inst:
        print ("index out of range")
        print inst
        for rot_pt in rot_points:               
            x = int(((rot_pt[0] + r) / (2 * r))*(patch_dim - 1))
            y = int(((rot_pt[1] + r) / (2 * r))*(patch_dim - 1))
            z = int(((rot_pt[2] + rz) / (2 * rz))*(patch_dim - 1))
            if (z >= 0) and (z < patch_dim) and (y >= 0) and (y < patch_dim) and (x >= 0) and (x < patch_dim):
                X1[x, y, z, 0] = 1
        #np.save(dir_temp + "sample_" + str(global_i) + "_" + str(aug_num), X1)
    except Exception as inst:
        print ("Unexpected exception: ", type(inst))
        print(inst.args)
        print(inst)
        #print ("Exception message: ", inst)
        raise
        """
        
    if(use_tsdf):
        X_tsdf = np.zeros((patch_dim, patch_dim, patch_dim, 1), np.float)
        X_tsdf[:, :, :, 0] = ndimage.morphology.distance_transform_edt(1 - X1[:, :, :, 0])
        X_tsdf[:, :, :, 0] /= np.max(X_tsdf[:, :, :, 0])
        np.save(dir_temp + "sample_" + str(global_i) + "_" + str(aug_num), X_tsdf)
    else:
        np.save(dir_temp + "sample_" + str(global_i) + "_" + str(aug_num), X1)    
    #return X[aug_num]
    

