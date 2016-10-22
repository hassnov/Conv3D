import numpy as np
from enum import Enum
import subprocess
import os.path
from plyfile import PlyData
import utils
from scipy import spatial


class SampleAlgorithm(Enum):
    Uniform = 1
    Random = 2
    ISS_Detector = 3

class Sampler:

#algorithm = SampleAlgorithm.Sampling_Uniform
    def __init__(self):
        print ''

    @staticmethod
    def sample_random(pc, num_point):
        indices = np.random.permutation(pc.shape[0])[0:num_point]
        sampled_pc = pc[indices]
        return sampled_pc, indices
    
    @staticmethod
    def sample_uniform(pc, sample_step):
        indices = np.arange(0, pc.shape[0]-1, int(sample_step))
        sampled_pc = pc[indices]
        return sampled_pc, indices

    @staticmethod
    def sample_ISS(file_name, min_num_point, pose):
        root, ext = os.path.splitext(file_name)
        in_file = root + ".ply"
        out_file = root + "_iss.ply"
        if (not os.path.isfile(out_file)):
            print "file doen't exits.................."
            args = ["./iss_detect", in_file, out_file]
            popen = subprocess.Popen(args, stdout=subprocess.PIPE)
            popen.wait()
            output = popen.stdout.read()
            print output
        pc = np.load(root + '.npy')
        tree = spatial.KDTree(pc)
        ply = PlyData.read(out_file)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        pc_iss = np.asarray(zip(x.ravel(), y.ravel(), z.ravel()))
        indices = np.zeros((pc_iss.shape[0],))
        for pt_i, samplept in enumerate(pc_iss):
            _, index = tree.query(samplept, k=1)
            indices[pt_i] = index    
        pc_iss = utils.transform_pc(pc_iss, pose)
        
        #min_num_point = min(int(pc_iss.shape[0] / 10), 200)
        if min_num_point >= pc_iss.shape[0]:
            return pc_iss, indices
        
        sample_step = int(pc_iss.shape[0] / min_num_point)
        pc_iss_samples, _ = Sampler.sample_uniform(pc_iss, sample_step)
        indices_samples, _ = Sampler.sample_uniform(indices, sample_step)
        assert(pc_iss_samples.shape[0] == indices_samples.shape[0])
        print ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,pc_iss shape:", pc_iss_samples.shape
        return pc_iss_samples, indices_samples
        #return Sampler.sample_uniform(pc_iss, sample_step)[0], np.full([min_num_point,], -1)
        
        #indices = np.arange(0, pc.shape[0]-1, int(sample_step))
        #sampled_pc = pc[indices]
        #return sampled_pc, indices


    @staticmethod
    def sample(pc, rel_Distance, min_num_point, file_name="points/points.npy",
               pose = np.identity(4),
               sampling_algorithm=SampleAlgorithm.Uniform):
        if sampling_algorithm == SampleAlgorithm.Uniform:
            return Sampler.sample_uniform(pc, int(pc.shape[0] / min_num_point))
        else:
            if sampling_algorithm == SampleAlgorithm.Random:
                return Sampler.sample_random(pc, min_num_point)
            if sampling_algorithm == SampleAlgorithm.ISS_Detector:
                return Sampler.sample_ISS(file_name, min_num_point, pose=pose)                
        return -1

