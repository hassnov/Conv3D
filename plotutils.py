import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
import numpy as np
import utils
import itertools

def plot_patch_3D(patch, name='Patch'):
    x = []
    y = []
    z = []
    for xi in np.arange(0, patch.shape[0]):
        for yi in np.arange(0, patch.shape[1]):
            for zi in np.arange(0, patch.shape[2]):
                if patch[xi, yi, zi] >= 1:
                    x.append(xi)
                    y.append(yi)
                    z.append(zi)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z)
    plt.title(name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return fig


def show_pc(pc, points_in_pc=[], mode='point'):
    pts = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(1, 1, 1), mode='point')
    
    if len(points_in_pc) > 0:
        pts = mlab.points3d(points_in_pc[:, 0], points_in_pc[:, 1], points_in_pc[:, 2],np.full(points_in_pc[:, 2].shape, 1, dtype='i8'),
                      color=(0, 0, 0), colormap='Spectral', mode=mode, scale_factor=0.001)
        pts.glyph.glyph.clamping = False
    


def show_matches(pc1, pc2, samples1, samples2, matches, tube_radius=0.0003, N=50):
    axis = 2
    pc1[:, axis] -= 0.1
    samples1[:, axis] -= 0.1
    pc2[:, axis] += 0.1
    samples2[:, axis] += 0.1
    show_pc(pc1, pc2)
    assert(samples1.shape[0] == samples2.shape[0] == matches.shape[0])
    if matches.shape[0] > N:
        samples11 = samples1[matches[0:50,0].astype(dtype=np.int16)]
        samples22 = samples2[matches[0:50,1].astype(dtype=np.int16)]
    else:
        samples11 = samples1[matches[:,0].astype(dtype=np.int16)]
        samples22 = samples2[matches[:,1].astype(dtype=np.int16)]
    assert (samples11.shape == samples22.shape)
    line = np.zeros((2, 3))
    i = 0
    correct = 0
    wrong = 0
    for sample1, sample2 in zip(samples11, samples22):
        line[0] = sample1
        line[1] = sample2
        if matches[i,0] == matches[i, 1]:
            mlab.plot3d(line[:, 0], line[:, 1], line[:, 2], color=(0, 1, 0), tube_radius=tube_radius)
            correct += 1
        else:
            mlab.plot3d(line[:, 0], line[:, 1], line[:, 2], color=(1, 0, 0), tube_radius=tube_radius)
            wrong += 1
        i += 1
    print 'correct: ', float(correct) / (correct + wrong), '    worng: ', float(wrong) / (correct + wrong)
    mlab.show()
    return -1

def draw_normal(mu, nr, tube_radius=0.0003):
    nrr = np.array(utils.normalize(nr))
    line = np.zeros((2, 3))
    line[0] = mu
    line[1] = mu - 0.02*nrr
    mlab.plot3d(line[:, 0], line[:, 1], line[:, 2], color=(1, 0, 0), tube_radius=tube_radius)
    

