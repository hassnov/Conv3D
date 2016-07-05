import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab
import numpy as np
import utils

def plot_patch(patch):
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
    plt.title('Patch')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax


def show_pc(pc, points_in_pc=[], mode='point'):
    pts = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(1, 1, 1), mode='point')
    
    if len(points_in_pc) > 0:
        pts = mlab.points3d(points_in_pc[:, 0], points_in_pc[:, 1], points_in_pc[:, 2],np.full(points_in_pc[:, 2].shape, 1, dtype='i8'),
                      color=(0, 0, 0), colormap='Spectral', mode=mode, scale_factor=0.001)
        pts.glyph.glyph.clamping = False
    

def draw_normal(mu, nr, tube_radius=0.0003):
    nrr = np.array(utils.normalize(nr))
    line = np.zeros((2, 3))
    line[0] = mu
    line[1] = mu - 0.02*nrr
    mlab.plot3d(line[:, 0], line[:, 1], line[:, 2], color=(1, 0, 0), tube_radius=tube_radius)
    

