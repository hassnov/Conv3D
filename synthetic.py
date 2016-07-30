import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from scipy import spatial
import plotutils
import math

def test():
    print 'test'
    
    rangesmin = np.arange(-1.5, -0.5, 0.05)
    rangesmax = np.arange(0.5, 1.5, 0.05)
    ranges = zip(rangesmin, rangesmax)
    print ranges
    nx = 100
    ny = 100
    
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
        
        #medx = xrng[xrng.shape[0]/2 - 1]
        #medy = xrng[xrng.shape[0]/2 - 1]
        r = np.min( [(maxrangex - minrangex) / float(2), (maxrangey - minrangey) / float(2)])
        
        midpoint = [medx, medy, 0]
        print 'mid: ', midpoint
        i = tree.query_ball_point(midpoint, r=r)
        _, indices = tree.query(midpoint, k=len(i))
        local_points = points[indices]
        
        #plotutils.show_pc(points, local_points)
        #mlab.show()
        
        
        ###################################################################
        x = local_points[:, 0]
        y = local_points[:, 1]
        
        seg = 0.4
        seg2 = seg**2
        seg4 = seg**4
        
        zhat = (1/(math.pi*seg4))*(1 - (x**2 + y**2)/(2*seg2))*np.exp(-(x**2 + y**2)/(2*seg2))#*math.exp(-(x**2 + y**2)/(2*seg2))
        z = np.sin(x**2 + y**2) / (x**2 + y**2)
        z = x**2 + y**2
        #z = (x**2 - y**2)
        z = zhat/10
        
        a = 2
        b = -1
        s = 1/float(3)
        z = s*(a*x**2 + b*y**2)
        
        mlab.points3d(x, y, z, color=(1, 1, 1), mode='point')
        mlab.show()
        
        print 'xshape: ', x.shape
        print 'yshape: ', y.shape
        print 'zshape: ', z.shape


test()