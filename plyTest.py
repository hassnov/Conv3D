import numpy as np
from numpy import linalg as LA, dtype
from plyfile import PlyData
from scipy import spatial
from mayavi import mlab
import os.path
import matplotlib.pyplot as plt
from sampling import Sampler, SampleAlgorithm
import utils
import time
from mpl_toolkits.mplot3d import Axes3D


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
    show_pc(sample_points)
    mlab.show()
    show_pc(pc)
    mlab.show()
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
            xs = []
            ys = []
            zs = []
            for rot_pt in rot_points:
                #x = int(patch_dim*(rot_pt[0] / (2*l)) + patch_dim / 2)
                #y = int(patch_dim*(rot_pt[1] / (2*l)) + patch_dim / 2)
                #z = int(patch_dim*(rot_pt[2] / (2*l)) + patch_dim / 2)
                x = int(((rot_pt[0] + l) / (2 * l))*(patch_dim - 1))
                y = int(((rot_pt[1] + l) / (2 * l))*(patch_dim - 1))
                z = int(((rot_pt[2] + l) / (2 * l))*(patch_dim - 1))
                if (z >= 0) and (z < patch_dim) and (y >= 0) and (y < patch_dim) and (x >= 0) and (x < patch_dim):
                    patch[x, y, z] = 1
                    if (x not in xs) or (y not in ys) or (z not in zs):
                        xs.append(int(x))
                        ys.append(int(y))
                        zs.append(int(z))
                else:
                    print 'rot_pt[0] / (l*2) = ', rot_pt[0] / (r*2), '    rot_pt[1] / (l*2) = ',  rot_pt[1] / (r*2)
                
            patches.append(patch)
                                 
#header = 'patch: ' + str(point_number) + ', rotation: ' + str(theta)
#np.savetxt('E:/Fac/BMC Master/Thesis/Models/bunny/reconstruction/plytest/patchesPython/patch_' +
#          str(point_number), patch, delimiter=',', header=header)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(xs, ys, zs)
        plt.title('Patch')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
            #show_pc(patch)
        #print 'patches ', len(patches)
        #np.save('patches/patches_' + str(point_number), np.array (patches))
    return patches


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
    sample_points = Sampler.sample(pc, -1, 1000)
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
        

main_patch()
