import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
import numpy as np
import utils
import itertools

def plot_patch_3D(patch, name='Patch', fig = -1):
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
    
    if fig == -1:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z)
    plt.title(name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #TODO: general size
    ax.set_xlim([0,31])
    ax.set_ylim([0,31])
    ax.set_zlim([0,31])
    
    return fig

def plot_patch_TSDF(patch, cutoff=0.1, name='Patch'):
    x = []
    y = []
    z = []
    s = []
    c = []
    sizex = patch.shape[0]
    sizey = patch.shape[1]
    sizez = patch.shape[2]
    print 'size: ', patch.shape
    for xi in np.arange(0, sizex):
        for yi in np.arange(0, sizey):
            for zi in np.arange(0, sizez):
                if patch[xi, yi, zi] <= cutoff:
                    x.append(xi/float(1000))
                    y.append(yi/float(1000))
                    z.append(zi/float(1000))
                    s.append(cutoff - patch[xi, yi, zi, 0])
                    #s.append(0.1)
    
    #s = np.reshape(patch, [sizex*sizey*sizez]) / np.max(patch)
    #print patch
    print 'smax: ', np.max(s)
    print "tsdf points: ", len(x)
    tsdf = mlab.points3d(x, y, z, s, colormap='Spectral', scale_factor=0.004, scale_mode="scalar")
    tsdf.module_manager.scalar_lut_manager.reverse_lut = True
    return tsdf

    return mlab.points3d(x, y, z, s, colormap='Spectral', scale_factor=0.004, scale_mode="scalar")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z, c=np.reshape(patch, [sizex*sizey*sizez]))
    plt.title(name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0,sizex-1])
    ax.set_ylim([0,sizey-1])
    ax.set_zlim([0,sizez-1])
    
    return fig

def show_pc(pc, points_in_pc=[], mode='point', scale_factor = 0.001):
    pts = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(1, 1, 1), mode='point')
    
    if len(points_in_pc) > 0:
        pts = mlab.points3d(points_in_pc[:, 0], points_in_pc[:, 1], points_in_pc[:, 2],np.full(points_in_pc[:, 2].shape, 1, dtype='i8'),
                      color=(0, 0, 0), colormap='Spectral', mode=mode, scale_factor=scale_factor)
        pts.glyph.glyph.clamping = False
    


def show_matches(pc1, pc2, samples1, samples2, matches, tube_radius=0.0003, N=50):
    axis = 0
    diff = 1
    bbx = utils.compute_bounding_box_std(pc1)
    pc1[:, axis] -= (bbx[axis][1] - bbx[axis][0])*1.5
    samples1[:, axis] -= (bbx[axis][1] - bbx[axis][0])*1.5
    pc2[:, axis] += (bbx[axis][1] - bbx[axis][0])*1.5
    samples2[:, axis] += (bbx[axis][1] - bbx[axis][0])*1.5
    show_pc(pc1, pc2)
    assert(samples1.shape[0] == samples2.shape[0] == matches.shape[0])
    if matches.shape[0] > N:
        samples11 = samples1[matches[0:N,0].astype(dtype=np.int16)]
        samples22 = samples2[matches[0:N,1].astype(dtype=np.int16)]
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
    

