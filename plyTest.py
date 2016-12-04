import numpy as np
from numpy import linalg as LA
from scipy import spatial
from mayavi import mlab
import os.path
import matplotlib.pyplot as plt
from sampling import Sampler, SampleAlgorithm
import time
from mpl_toolkits.mplot3d import Axes3D

import plotutils
import utils


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


def create_ref_points_files(pc, sample_points, l, path='ref_points/', use_normals_pc=True, use_point_as_mean=False, flip_view_point=False, sigma=0.7071):
    r = l / np.sqrt(2)
    print 'r= ', r
    tree = spatial.KDTree(pc)
    for point_number, samplept in enumerate(sample_points):
        start_time = time.time()
        i = tree.query_ball_point(samplept[0:3], r=r)
        distances, indices = tree.query(samplept[0:3], k=len(i))
        local_points = pc[indices]
        if len(i) <= 8:
            continue
        if use_point_as_mean:
            mu = samplept[0:3]
        else:
#TODO: compute real mean
            mu = samplept[0:3]
        if (use_normals_pc) and (samplept.shape == 6):
            nr = samplept[3:6]
#TODO: get_weighted_normal
        else:# calc normals
            cov_mat = utils.build_cov(local_points, mean=mu)
            w, v = LA.eigh(cov_mat)
            min_id = np.argmin(w)
            nr = np.transpose(v[:, min_id])
            nr = utils.normalize(nr)
        
# at this point the plane is defined by (mu, nr)
# we transform the mean to be (0,0,0) and the normal to be  (0,0,1)
# to obtain canonical frame
        z_axis = np.array([0, 0, 1])
        origin = np.zeros(3)
        local_pose = utils.align_vectors(mu, nr, origin, z_axis)
        ref_points = utils.transform_pc(local_points, pose=local_pose)
        fname = 'ref_' + str(point_number)
        fname = os.path.join(path, fname)
        #np.save(fname, ref_points)
        duration = time.time() - start_time
        print 'duration: ', duration
        

def extract_patches_vox(pc, sample_points, l, num_rotations=20, patch_dim=24, use_normals_pc=True, use_point_as_mean=False, flip_view_point=False, sigma=0.7071):
    r = l / np.sqrt(2)
    print 'r= ', r
    print 'l=', l,'    r=', r
    tree = spatial.KDTree(pc)
    patches = []
    #show_pc(sample_points)
    #mlab.show()
    #show_pc(pc)
    #mlab.show()
    for point_number, samplept in enumerate(sample_points):
        patches = []
        i = tree.query_ball_point(samplept[0:3], r=r)
        distances, indices = tree.query(samplept[0:3], k=len(i))
        local_points = pc[indices]
        if len(i) <= 8:
            continue
        if use_point_as_mean:
            mu = samplept[0:3]
        else:
            #TODO: compute real mean
            mu = samplept[0:3]
        if (use_normals_pc) and (samplept.shape == 6):
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
        #np.save('patches/ref_' + str(point_number), ref_points)
        #print 'ref points: ', ref_points.shape
        
        print 'point number: ', point_number
        show_pc(pc, local_points, mode='sphere')
        draw_normal(mu, nr, tube_radius=0.0005)
        mlab.show()
        
        show_pc(ref_points, np.array([0, 0, 0]).reshape((1, 3)))
        draw_normal(origin, z_axis)
        mlab.show()
        for theta in utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations):
            rot2d = utils.angle_axis_to_rotation(np.cos(theta), z_axis)
            rot_points = utils.transform_pc(ref_points, rot2d)
            print 'rot_max: ', rot_points[:, 0].max()
            patch = np.zeros((patch_dim, patch_dim, patch_dim), dtype='int32')
            #xs = []
            #ys = []
            #zs = []
            for rot_pt in rot_points:
                #x = int(patch_dim*(rot_pt[0] / (2*l)) + patch_dim / 2)
                #y = int(patch_dim*(rot_pt[1] / (2*l)) + patch_dim / 2)
                #z = int(patch_dim*(rot_pt[2] / (2*l)) + patch_dim / 2)
                x = int(((rot_pt[0] + l) / (2 * l))*(patch_dim - 1))
                y = int(((rot_pt[1] + l) / (2 * l))*(patch_dim - 1))
                z = int(((rot_pt[2] + l) / (2 * l))*(patch_dim - 1))
                if (z >= 0) and (z < patch_dim) and (y >= 0) and (y < patch_dim) and (x >= 0) and (x < patch_dim):
                    patch[x, y, z] = 1
                    #if (x not in xs) or (y not in ys) or (z not in zs):
                     #   xs.append(int(x))
                     #   ys.append(int(y))
                      #  zs.append(int(z))
                else:
                    print 'rot_pt[0] / (l*2) = ', rot_pt[0] / (r*2), '    rot_pt[1] / (l*2) = ',  rot_pt[1] / (r*2)
            plot_patch(patch)
            plt.show()
            patches.append(patch)
                                 
#header = 'patch: ' + str(point_number) + ', rotation: ' + str(theta)
#np.savetxt('E:/Fac/BMC Master/Thesis/Models/bunny/reconstruction/plytest/patchesPython/patch_' +
#          str(point_number), patch, delimiter=',', header=header)
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter3D(xs, ys, zs)
        #plt.title('Patch')
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        #plt.show()
            #show_pc(patch)
        #print 'patches ', len(patches)
        #np.save('patches/patches_' + str(point_number), np.array (patches))
    return patches


def extract_patches_2d(pc, sample_points, l, num_rotations=20, patch_dim=24, use_normals_pc=True, use_point_as_mean=False, flip_view_point=False, sigma=0.7071):
    r = l / np.sqrt(2)
    print 'r= ', r
    print 'l=', l,'    r=', r
    tree = spatial.KDTree(pc)
    patches = []
    #show_pc(sample_points)
    #mlab.show()
    #show_pc(pc)
    #mlab.show()
    for point_number, samplept in enumerate(sample_points):
        patches = []
        i = tree.query_ball_point(samplept[0:3], r=r)
        distances, indices = tree.query(samplept[0:3], k=len(i))
        local_points = pc[indices]
        if len(i) <= 8:
            continue
        if use_point_as_mean:
            mu = samplept[0:3]
        else:
            #TODO: compute real mean
            mu = samplept[0:3]
        if (use_normals_pc) and (samplept.shape == 6):
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
        #np.save('patches/ref_' + str(point_number), ref_points)
        #print 'ref points: ', ref_points.shape
        
        print 'point number: ', point_number
        show_pc(pc, local_points, mode='sphere')
        draw_normal(mu, nr, tube_radius=0.0005)
        mlab.show()
        
        show_pc(ref_points, np.array([0, 0, 0]).reshape((1, 3)))
        draw_normal(origin, z_axis)
        mlab.show()
        for theta in utils.my_range(-np.pi, np.pi, (2*np.pi)/num_rotations):
            rot2d = utils.angle_axis_to_rotation(np.cos(theta), z_axis)
            rot_points = utils.transform_pc(ref_points, rot2d)
            print 'rot_max: ', rot_points[:, 0].max()
            patch = np.zeros((patch_dim, patch_dim, 3), dtype='int32')
            #xs = []
            #ys = []
            #zs = []
            for rot_pt in rot_points:
                #project on z
                x = int(patch_dim*(rot_pt[0] / l) + patch_dim / 2)
                y = int(patch_dim*(rot_pt[1] / l) + patch_dim / 2)
                if (y >= 0) and (y < patch_dim) and (x >= 0) and (x < patch_dim):
                    patch[x, y, 2] = 1 # rot_pt[2]
                
                #project on y
                x = int(patch_dim*(rot_pt[0] / l) + patch_dim / 2)
                z = int(patch_dim*(rot_pt[2] / l) + patch_dim / 2)
                if (z >= 0) and (z < patch_dim) and (x >= 0) and (x < patch_dim):
                    patch[z, x, 1] = 1 #rot_pt[1]
                
                #project on x
                y = int(patch_dim*(rot_pt[1] / l) + patch_dim / 2)
                z = int(patch_dim*(rot_pt[2] / l) + patch_dim / 2)
                if (z >= 0) and (z < patch_dim) and (y >= 0) and (y < patch_dim):
                    patch[z, y, 0] = 1 #rot_pt[0]
                plot_patch(patch)
                plt.show()
            patches.append(patch)
                                 
#header = 'patch: ' + str(point_number) + ', rotation: ' + str(theta)
#np.savetxt('E:/Fac/BMC Master/Thesis/Models/bunny/reconstruction/plytest/patchesPython/patch_' +
#          str(point_number), patch, delimiter=',', header=header)
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter3D(xs, ys, zs)
        #plt.title('Patch')
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        #plt.show()
            #show_pc(patch)
        #print 'patches ', len(patches)
        #np.save('patches/patches_' + str(point_number), np.array (patches))
    return patches



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

import random
def add_noise(pc, prob=0.3):
    pc1 = np.zeros(pc.shape)
    pp = random.Random()
    for i, point in enumerate(pc):
        if pp.random() < prob:
            pc1[i] = add_noise_point(point)
        else:
            pc1[i] = point
    return pc1

def add_noise_point(pt):
    pt1 = np.zeros(pt.shape)
    pt1[0] = pt[0] + pt[0]*random.Random().random()*0.03
    pt1[1] = pt[1] + pt[1]*random.Random().random()*0.03
    pt1[2] = pt[2] + pt[2]*random.Random().random()*0.03
    return  pt1

def plot(ply):
    vertex = ply['vertex']
    (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))

    mlab.points3d(x, y, z, color=(1, 1, 1), mode='point')

    if 'face' in ply:
        tri_idx = ply['face']['vertex_indices']
        idx_dtype = tri_idx[0].dtype

        triangles = np.fromiter(tri_idx, [('data', idx_dtype, (3,))],
                                   count=len(tri_idx))['data']

        mlab.triangular_mesh(x, y, z, triangles,
                             color=(1, 0, 0.4), opacity=0.5)


def main():
    sample_points = utils.read_ply('/media/hasan/DATA/Fac/BMC Master/Thesis/Models/bunny/reconstruction/plytest/bun_zipper_sampled.ply')
    pc = utils.read_ply('/media/hasan/DATA/Fac/BMC Master/Thesis/Models/bunny/reconstruction/plytest/bun_zipper.ply')
    pc_diameter = utils.get_pc_diameter(pc)
    relL = 0.07
    l = relL*pc_diameter
    create_ref_points_files(pc, sample_points, l)
    
def main_sample():
    pc = utils.read_ply('E:/Fac/BMC Master/Thesis/Models/bunny/reconstruction/plytest/bun_zipper.ply')
    sample_points = Sampler.sample(pc, -1, 1000, sampling_algorithm=SampleAlgorithm.Uniform)
    mlab.points3d(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], color=(1, 1, 1), mode='point')
    mlab.show()
    print 'done'
    
import os

def main_patch():
    dir1 = os.path.dirname(os.path.realpath(__file__))
    sample_points = utils.read_ply(os.path.join(dir1, 'plytest/bun_zipper_sampled.ply'))
    pc = utils.read_ply(os.path.join(dir1, 'plytest/bun_zipper.ply'))
    num_rotations = 20
    patch_dim = 32
    relSampling = 0.05
    relRadius = 0.1
    pc_diameter = utils.get_pc_diameter(pc)
    radius = pc_diameter*relRadius
    relL = 0.07
    l = relL*pc_diameter
    extract_patches_vox(pc, sample_points, l=l, num_rotations=num_rotations, patch_dim=patch_dim)


def main_noise():
    #pc = utils.read_ply('/media/hasan/DATA/Fac/BMC Master/Thesis/Models/bunny/reconstruction/plytest/bun_zipper_sampled.ply')
    pc = utils.read_ply('/media/hasan/DATA/Fac/BMC Master/Thesis/Models/bunny/reconstruction/plytest/bun_zipper.ply')
    noisy = add_noise(pc, prob=0.3)
    
    plotutils.show_pc(pc)
    mlab.show()
    
    plotutils.show_pc(noisy)
    mlab.show()

def main_show_ply():
    
    
    for i in range(0, 540, 45):
        patch = np.load("/home/hasan/workspace/Conv3D/temp_9/sample_1000_{0}.npy".format(i))
    #recon = np.load('/home/hasan/workspace/Conv3D/temp_93/recon/sample_1111_179.npy')
        plotutils.plot_patch_3D(patch)
        plt.show()
    
    return 0
    for i in range(18):
        #pc = np.load("/home/hasan/workspace/Conv3D/scenes/Scene{0}_0.1.ply".format(i))
        pc = utils.read_ply("/home/hasan/workspace/Conv3D/scenes/Scene{0}_0.1.ply".format(i))
        mr = utils.model_resolution(pc)
        print "shape: ", pc.shape, "    mr: ", mr
        plotutils.show_pc(pc)
        
        
    return 0
    

    
    
    return 0
    
    for i in range(0, 360, 60):
        sample = 'sample_4817_' + str(i)
        patch = np.load('/home/hasan/workspace/Conv3D/temp_93/' + sample + '.npy')
        plotutils.plot_patch_3D(patch)
        plt.show()
    return 0
    #for i in range(100, 112):
    #pc = utils.read_ply('/home/hasan/workspace/Conv3D/temp_7500/ref_points/sample_99.npy')
    for i in range(0, 6400, 100):
        pc = np.load('/home/hasan/workspace/Conv3D/temp_7500/ref_points/sample_' + str(i) + '.npy')
        plotutils.show_pc(pc)
        mlab.show()
    #samples, _ = Sampler.sample(pc, -1, 100000, "", pose=-1, sampling_algorithm=SampleAlgorithm.Uniform)
    #np.save("points/lucysample", samples)
    
    
    return 0 

    
    patch = np.load("temp/sample_5_1.npy")
    plotutils.plot_patch_3D(patch)
    plt.show()
    
    
import cluster_points
def main_show_cluster():
    dir_temp = "/media/hasan/DATA/shapenet/ref_points/"
    dir_temp = "/home/hasan/workspace/Conv3D/temp_1100/ref_points/"
    reader = cluster_points.ClusterPoints()
    reader.load_dataset(dir_temp, bandwidth=0.00001)
    print reader.labels
    group = np.argwhere(reader.labels == 3)
    group = group.reshape([group.shape[0]])
    print "samples: ", group.shape[0]
    for i in group:
        pc = np.load(dir_temp + "sample_" + str(i) + ".npy")
        plotutils.show_pc(pc)
        mlab.show()
    print group
    return 0



#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import NullFormatter
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from sklearn import manifold
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
import cv2

def cluster_list(eigen_list, bandwidth=0.00015):
    
    
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(eigen_list)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    return labels, n_clusters_


def cluster_list_kmeans(eigen_list, k):
    km = KMeans(k).fit(eigen_list)
    return km.labels_, []

def main_tsne():
    fpfh_list = np.load("/home/hasan/workspace/Conv3D/temp_32/eigen_list.npy")
    #codes = np.load("temp_32/codes")
    #labels = np.load("labels.npy")
    #labels, _ = cluster_list(fpfh_list, bandwidth=0.001)
    #labels, _ = cluster_list_kmeans(fpfh_list, k=300)
    labels = np.load("temp_32/labels_train1000.npy")
    #np.save("labels", labels)
    
    t0 = time.time()
    print "tsne running..."
    #Y = np.load("tsne.npy")
    #print Y
    
    model = manifold.TSNE(n_components=2, random_state=0)
    Y = model.fit_transform(fpfh_list)
    #np.save("tsne", Y)
    t = time.time() - t0
    print("t-SNE: %.2g sec" % t)
    
    #fig = plt.gcf()
    fig = plt.Figure()
    fig.clf()
    ax = plt.subplot(111)
    for sample_i, pt in enumerate(Y):
        if labels[sample_i] != 0:
            continue
        pix = plt.imread("/media/hasan/DATA1/shapenet/temp_32/screenshots/sample_" + str(sample_i) + ".png")
        #cv2.imshow("sample", pix)
        #cv2.waitKey()
        imagebox = OffsetImage(pix, zoom=0.25)
        xy = pt
        ab = AnnotationBbox(imagebox, xy, xycoords='data', frameon=False)                                  
        ax.add_artist(ab)
        #off = 1
        #plt.imshow(pix, extent=[xy[0] - off, xy[0] + off, xy[1] - off, xy[1] + off])        
    
    ax.grid(True)
    ax.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.draw()
    #plt.savefig("plots/tsne.pdf", format='pdf')
    plt.show()
    return 0

    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.axis('tight')
    plt.show()
    return 0    

def autoencoder_result():
    i = 0
    sample = np.load("/home/hasan/workspace/Conv3D/temp_1100/sample_" + str(i) + "_12.npy")
    recon = np.load("/home/hasan/workspace/Conv3D/temp_1100/recon/sample_" + str(i) + "_12.npy")
    recon [recon > 0.5] = 1
    recon [recon <= 0.5] = 0
    print np.sum(np.sqrt(np.abs(recon-sample)))
    plotutils.plot_patch_3D(sample, "sample")
    plotutils.plot_patch_3D(recon, "recon")
    plt.show()
    

def save_clusters_images():
    #num_samples = 3443
    for i in range(3443):
        sample = np.load("/home/hasan/workspace/Conv3D/temp_32/ref_points/sample_" + str(i) + ".npy")
        plotutils.show_pc(sample)
        mlab.savefig("/media/hasan/DATA1/shapenet/temp_32/screenshots/sample_" + str(i) + ".png", size=[200,200], figure=mlab.gcf())
        #mlab.savefig("/home/hasan/workspace/Conv3D/temp_32/screenshots/sample_" + str(i) + ".png", size=[200,200], figure=mlab.gcf())
        mlab.clf()
        print "sample: ", i
        #mlab.show()
    return 0 


def show_tsdf():
    for i in range(10):
        patch = np.load("/home/hasan/workspace/Conv3D/temp_5tsdf/sample_{0}_0.npy".format(i))
        #recon = np.load('/home/hasan/workspace/Conv3D/temp_93/recon/sample_1111_179.npy')
        plotutils.plot_patch_TSDF(patch, cutoff=0.2, name='tsdf')
        mlab.show()

main_show_ply()
    
        