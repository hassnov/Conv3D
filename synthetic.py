import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from scipy import spatial
import plotutils
import math
import utils
from numpy import linalg as LA
from itertools import chain

def create_files():
    print 'create_files'
    
    rangesmin = np.arange(-1.0, -0.09, 0.3) / 100
    rangesmax = np.arange(1.0, 1.91, 0.3) / 100
    ranges = zip(rangesmin, rangesmax)
    #print ranges
    nx = 40
    ny = 40
    iglobal = 0
    
    for range1 in ranges:
        minrange = range1[0]
        maxrange = range1[1]
        #minrange = -1
        #maxrange = 1.0
        
        #Could be different for x and y
        minrangex = minrange
        maxrangex = maxrange
        minrangey = minrange
        maxrangey = maxrange
        xrng = np.linspace(minrangex, maxrangex/float(1), nx)
        yrng = np.linspace(minrangey, maxrangey/float(1), ny)
        
        x, y = np.meshgrid(xrng, yrng)
        
        x = np.reshape(x, x.shape[0]*x.shape[1])
        y = np.reshape(y, y.shape[0]*y.shape[1])
        z = np.zeros(x.shape)
        points = zip(x.ravel(), y.ravel(), z.ravel())
        points = np.asarray(points)
        tree = spatial.KDTree(points) 
        
        medx = minrangex + (maxrangex - minrangex)/float(2)
        medy = minrangey + (maxrangey - minrangey)/float(2)
        
        r = np.min( [(maxrangex - minrangex) / float(2), (maxrangey - minrangey) / float(2)])
        
        midpoint = [medx, medy, 0]
        print 'mid: ', midpoint
        i = tree.query_ball_point(midpoint, r=r)
        _, indices = tree.query(midpoint, k=len(i))
        local_points_surface = points[indices]    
        
        ###################################################################
        
        
        x = local_points_surface[:, 0]
        y = local_points_surface[:, 1]
        
        seg = 0.4
        seg2 = seg**2
        seg4 = seg**4
        zhat = 0.1 * ((1/(math.pi*seg4))*(1 - (x**2 + y**2)/(2*seg2))*np.exp(-(x**2 + y**2)/(2*seg2)))#*math.exp(-(x**2 + y**2)/(2*seg2))
        z = np.sin(x**2 + y**2) / (x**2 + y**2)
        z = x**2 + y**2
        
        a = 2
        b = -1
        s = 40.0
        
        concatinanted = chain(np.arange(-1, -0.39, 0.4), np.arange(0.4, 1.1, 0.4))
        arange = [-1, 1]
        brange = [1]
        #print concatinanted.next()
        for a in arange:
            for b in brange:
                
                print '(a,b) = ', a , ', ', b
                z = s*(a*x**2 + b*y**2)
                local_points = np.asarray (zip (x, y, z))
                """
                currMid = np.asarray([medx, medy, s*(a*medx**2 + b*medy**2)])
                currMid = np.reshape(currMid, [1,3])
                mlab.points3d(x, y, z, color=(1, 1, 1), mode='point')
                mlab.points3d(currMid[:, 0], currMid[:, 1], currMid[:, 2], np.full(currMid[:, 2].shape, 1, dtype='i8'), 
                              color=(0, 0, 0), mode='sphere', scale_factor=0.1)
                mlab.show()
                """
                
                point_count = x.shape[0]
                mu = np.zeros(3)
                mu[0] = np.sum(x) / point_count
                mu[1] = np.sum(y) / point_count
                mu[2] = np.sum(z) / point_count
                
                cov_mat = utils.build_cov(local_points, mean=mu)
                w, v = LA.eigh(cov_mat)
                min_id = np.argmin(w)
                #print 'eigenvalues: ', w
                nr = np.transpose(v[:, min_id])
                nr = utils.normalize(nr)
                z_axis = np.array([0, 0, 1])
                origin = np.zeros(3)
                
                local_pose = utils.align_vectors(mu, nr, origin, z_axis)
                ref_points = utils.transform_pc(local_points, pose=local_pose)
                
                X, _ = utils.get_patches(ref_points, 1, 1, 32)
                #plotutils.plot_patch_3D(X[0])
                #plt.show()
                plotutils.show_pc(ref_points, np.zeros((1, 3)), mode='sphere', scale_factor=0.001)
                mlab.show()
                
                
                #np.save('/media/hasan/DATA/Fac/BMC Master/Thesis/data_synthetic/sample_' + str(iglobal), ref_points)
                print 'sample ', iglobal
                iglobal += 1
        
        

create_files()

