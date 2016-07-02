import numpy as np
from numpy import linalg as LA
from plyfile import PlyData
from scipy import spatial


from sampling import Sampler, SampleAlgorithm
import utils


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


def angle_axis_to_rotation(c, axis):
    s = LA.norm(axis)
    
    vx = np.zeros((3, 3))
    vx[0, 1] = -axis[2]
    vx[0, 2] = axis[1]
    vx[1, 2] = -axis[0]
    vx[1, 0] = axis[2]
    vx[2, 0] = -axis[1]
    vx[2, 1] = axis[0]
    
    rotation = np.eye(3) + vx + ((1-c)/(s*s))*vx.dot(vx)
    return  rotation


def  align_vectors(p1, n1, p2, n2):
    axis = np.cross(n1, n2)
    c = np.dot(n1, n2)
    axis = normalize(axis)
    rotation = angle_axis_to_rotation(c, axis)
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
        

def augment(ref_points, l, num_rotations=20, patch_dim=24):
    patches = []
    z_axis = np.array([0, 0, 1])
    for theta in my_range(-np.pi, np.pi, (2*np.pi)/num_rotations):
        rot2d = utils.angle_axis_to_rotation(theta, z_axis)
        rot_points = utils.transform_pc(ref_points, rot2d)
        patch = np.zeros((patch_dim, patch_dim, patch_dim), dtype='int32')
        for rot_pt in rot_points:
            x = int(patch_dim*(rot_pt[0] / l) + patch_dim / 2)
            y = int(patch_dim*(rot_pt[1] / l) + patch_dim / 2)
            z = int(patch_dim*(rot_pt[2] / l) + patch_dim / 2)
            if (z >= 0) and (z < patch_dim) and (y >= 0) and (y < patch_dim) and (x >= 0) and (x < patch_dim):
                patch[x, y, z] = 1
        patches.append(patch)
     
