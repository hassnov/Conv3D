import numpy as np
from enum import Enum

class SampleAlgorithm(Enum):
    Uniform = 1
    Random = 2

class Sampler:

#algorithm = SampleAlgorithm.Sampling_Uniform
    def __init__(self):
        print ''

    @staticmethod
    def sample_random(pc, num_point):
        indices = np.random.permutation(pc.shape[0])[0:num_point, ...]
        sampled_pc = pc[indices]
        return sampled_pc, indices
    
    @staticmethod
    def sample_uniform(pc, sample_step):
        indices = np.arange(0, pc.shape[0]-1, int(sample_step))
        sampled_pc = pc[indices]
        return sampled_pc, indices

    @staticmethod
    def sample(pc, rel_Distance, min_num_point, sampling_algorithm=SampleAlgorithm.Uniform):
        if sampling_algorithm == SampleAlgorithm.Uniform:
            return Sampler.sample_uniform(pc, int(pc.shape[0] / min_num_point))
        else:
            if sampling_algorithm == SampleAlgorithm.Random:
                return Sampler.sample_random(pc, min_num_point)
        return -1

