import numpy as np
from numpy import linalg as LA
from plyfile import PlyData
import random
from scipy import spatial
import time

#from sampling import Sampler, SampleAlgorithm
#import utils


# Standard bounding box
def compute_bounding_box_std(pc):
    xrange = [pc[0, 0], pc[0, 0]]
    yrange = [pc[0, 1], pc[0, 1]]
    zrange = [pc[0, 2], pc[0, 2]]
    for pt in pc:
        if pt[0] < xrange[0]:
            xrange[0] = pt[0]
        if pt[0] > xrange[1]:
            xrange[1] = pt[0]

        if pt[1] < yrange[0]:
            yrange[0] = pt[1]
        if pt[1] > yrange[1]:
            yrange[1] = pt[1]

        if pt[2] < zrange[0]:
            zrange[0] = pt[2]
        if pt[2] > zrange[1]:
            zrange[1] = pt[2]

    return [xrange, yrange, zrange]


def get_pc_diameter(pc):
    xrange, yrange, zrange = compute_bounding_box_std(pc)
    dx = xrange[1] - xrange[0]
    dy = yrange[1] - yrange[0]
    dz = zrange[1] - zrange[0]
    return np.sqrt(dx*dx + dy*dy + dz*dz)


def read_ply(file_name):
    ply = PlyData.read(file_name)
    vertex = ply['vertex']
    (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
    points = zip(x.ravel(), y.ravel(), z.ravel())
    return np.asarray(points)


def build_cov(nb_pc, mean):
    accu = np.zeros((3, 3))
    for p in nb_pc:
        pt = (p - mean)
        accu += pt.reshape((3, 1)).dot(pt.reshape((1, 3)))

    points_count = nb_pc.shape[0]
    accu /= points_count
    return accu


def normalize(v):
    norm = LA.norm(v)
    if norm > 0:
        return [x / norm for x in v]
    return v


def angle_axis_to_rotation_old(angle, axis):
    c = np.cos(angle)
    s = LA.norm(axis)
    
    axis = normalize(axis)
    
    vx = np.zeros((3, 3))
    vx[0, 1] = -axis[2]
    vx[0, 2] = axis[1]
    vx[1, 2] = -axis[0]
    vx[1, 0] = axis[2]
    vx[2, 0] = -axis[1]
    vx[2, 1] = axis[0]
    
    rotation = np.eye(3) + vx + ((1-c)/(s*s))*vx.dot(vx)
    return  rotation

def rad(deg):
    return (deg*np.pi) / 180

def angle_axis_to_rotation(angle, axis):
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    axis1 = normalize(axis)
    
    m = np.zeros((3,3))
    m[0, 0] = c + axis1[0]*axis1[0]*t
    m[1, 1] = c + axis1[1]*axis1[1]*t
    m[2, 2] = c + axis1[2]*axis1[2]*t


    tmp1 = axis1[0]*axis1[1]*t
    tmp2 = axis1[2]*s
    m[1, 0] = tmp1 + tmp2
    m[0, 1] = tmp1 - tmp2
    tmp1 = axis1[0]*axis1[2]*t
    tmp2 = axis1[1]*s
    m[2, 0] = tmp1 - tmp2
    m[0, 2] = tmp1 + tmp2    
    tmp1 = axis1[1]*axis1[2]*t
    tmp2 = axis1[0]*s
    m[2, 1] = tmp1 + tmp2
    m[1, 2] = tmp1 - tmp2
    
    return  m


def  align_vectors(p1, n1, p2, n2):
    axis = np.cross(n1, n2)
    c = np.dot(n1, n2)
    axis = normalize(axis)
    rotation = angle_axis_to_rotation(np.arccos(c), axis)
    tr = np.zeros((3, 1))
    tr[0][0] = rotation[0, 0] * (-p1[0]) + rotation[0, 1] * (-p1[1]) + rotation[0, 2] * (-p1[2])
    tr[1][0] = rotation[1, 0] * (-p1[0]) + rotation[1, 1] * (-p1[1]) + rotation[1, 2] * (-p1[2])
    tr[2][0] = rotation[2, 0] * (-p1[0]) + rotation[2, 1] * (-p1[1]) + rotation[2, 2] * (-p1[2])
    temp = np.hstack((rotation, tr))
    return np.vstack((temp, np.array([0, 0, 0, 1])))


def transform_pc(pc, pose):
    if pose.shape == (3, 3):
        tr = np.zeros((3, 1))
        temp = np.hstack((pose, tr))
        pose = np.vstack((temp, np.array([0, 0, 0, 1])))
    elif pose.shape != (4, 4):
        return 0
    pct = np.zeros(pc.shape)
    for i, point in enumerate(pc):
        pt = np.append(point[0:3], 1)
        pt = pt.reshape((4, 1))
        pct[i][0:3] = pose.dot(pt)[0:3].reshape(3, )
#TODO: transform normals if exsited
    return pct


def calc_normal(nb_pc, mu):
    cov_mat = build_cov(nb_pc, mean=mu)
    w, v = LA.eigh(cov_mat)
    min_id = np.argmin(w)
    nr = np.transpose(v[:, min_id])
    return normalize(nr)


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step
  
  
def add_noise_old(pc, prob=0.3, factor=0.02):
    pc1 = np.zeros(pc.shape)
    pp = random.Random()
    for i, point in enumerate(pc):
        if pp.random() < prob:
            pc1[i] = add_noise_point(point, factor=factor)
        else:
            pc1[i] = point
    return pc1

def add_noise(pc, prob=0.3, factor=0.02):
    pc1 = pc #np.zeros(pc.shape)
    ii = np.random.permutation(pc.shape[0])[0 : int(pc.shape[0]*prob)]
    for i in ii:
        pc1[i] = add_noise_point(pc[i], factor=factor)
    return pc1


def add_noise_point(pt, factor=0.02):
    pt1 = np.zeros(pt.shape)
    pt1[0] = pt[0] + pt[0]*(random.Random().random())*factor
    pt1[1] = pt[1] + pt[1]*(random.Random().random())*factor
    pt1[2] = pt[2] + pt[2]*(random.Random().random())*factor
    return  pt1


def stack_matches(indices_orig, indices, distances):
    assert(len(indices) == len(distances))
    #ii = np.reshape( np.arange(0, len(indices)), [len(indices), 1])
    ii = np.reshape(np.asarray(indices_orig), [len(indices_orig), 1])
    ii_match = np.reshape(np.asarray(indices), [len(indices), 1])
    dd = np.reshape(np.asarray(distances), [len(indices), 1])
    
    i_d = np.hstack((ii, ii_match, dd))
    isort = np.argsort(dd[:, 0])
    i_d = i_d[isort]
    return i_d


def correct_matches(samples1, samples2, matches, tube_radius=0.0003, N=50, ignore_size=True):
    if(len(matches)==0):
        return 0, 0
    if not ignore_size:
        assert(samples1.shape[0] == samples2.shape[0] == matches.shape[0])
    assert(samples1.shape[0] == samples2.shape[0])
    matchespart = matches
    if matches.shape[0] > N:
        matchespart = matches[0:N,...]
        samples11 = samples1[matches[0:N,0].astype(dtype=np.int16)]
        samples22 = samples2[matches[0:N,1].astype(dtype=np.int16)]
    else:
        samples11 = samples1[matches[:,0].astype(dtype=np.int16)]
        samples22 = samples2[matches[:,1].astype(dtype=np.int16)]
    assert (samples11.shape == samples22.shape)
    i = 0
    correct = 0
    wrong = 0
    #for i in range(samples11.shape[0]):
    for match in matchespart:
        if match[0] == match[1]:
            correct += 1
        else:
            wrong += 1
        i += 1
#    print "correct: ", correct, "	Wrong: ", wrong, "	len(matches): ", len(matches)
    print 'correct: ', float(correct) / (correct + wrong), '    worng: ', float(wrong) / (correct + wrong)
    return float(correct) / (correct + wrong), float(wrong) / (correct + wrong)

def match_des(des1, des2, ratio=1):
    tree = spatial.KDTree(des1)
    counter = 0
    ii_orig = []
    ii = []
    dd = []
    for des in des2:
        d, i = tree.query(des, k=2)
        if abs(d[0] / d[1]) < ratio:
        #if True:
            ii_orig.append(counter)
            ii.append(i[0])
            dd.append(d[0])
        counter += 1
        
    
    matches = stack_matches(ii_orig, ii, dd)
    return matches


def get_patches(ref_points, num_rotations, lable, patch_dim):
    #xs, ys, zs = compute_bounding_box_std(ref_points)

    xs = [np.min(ref_points[:, 0]), np.max(ref_points[:, 0])]
    ys = [np.min(ref_points[:, 1]), np.max(ref_points[:, 1])]
    zs = [np.min(ref_points[:, 2]), np.max(ref_points[:, 2])]
    d = xs[1] - xs[0]
    if(xs[1] - xs[0] < ys[1] - ys[0]):
        d = ys[1] - ys[0]
    if (zs[1] - zs[0] > d):
        d = zs[1] - zs[0]
    r = d/float(2)
    X = np.zeros((num_rotations, patch_dim, patch_dim, patch_dim, 1), np.int32)
    Y = np.zeros((num_rotations), np.int32)
    z_axis = np.array([0, 0, 1])
    origin = np.zeros(3)
    for aug_num, theta in enumerate(my_range(-np.pi, np.pi, (2*np.pi)/num_rotations)):
        if aug_num == num_rotations:
            break
        rot2d = angle_axis_to_rotation(theta, z_axis)
        rot_points = transform_pc(ref_points, rot2d)
        #patch = np.zeros((self.patch_dim, self.patch_dim, self.patch_dim), dtype='int32')
        for rot_pt in rot_points:
            x = int(((rot_pt[0] + r) / (2 * r))*(patch_dim - 1))
            y = int(((rot_pt[1] + r) / (2 * r))*(patch_dim - 1))
            z = int(((rot_pt[2] + r) / (2 * r))*(patch_dim - 1))
            #x = int(self.patch_dim*(rot_pt[0] / self.l) + self.patch_dim / 2)
            #y = int(self.patch_dim*(rot_pt[1] / self.l) + self.patch_dim / 2)
            #z = int(self.patch_dim*(rot_pt[2] / self.l) + self.patch_dim / 2)
            if (z >= 0) and (z < patch_dim) and (y >= 0) and (y < patch_dim) and (x >= 0) and (x < patch_dim):
            #patch[x, y, z] = 1
            #X[point_number*num_rotations + aug_num, x + self.patch_dim * (y + self.patch_dim * z)] = 1
                X[aug_num, x, y, z, 0] = 1
        
        #X[point_number*num_rotations + aug_num, :] = patch.reshape((np.power(self.patch_dim, 3),))
        Y[aug_num] = lable
    return X, Y

