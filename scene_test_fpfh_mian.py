import ConfigParser
import os.path
import numpy
import utils
#import plotutils
#from mayavi import mlab
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plyfile import PlyData
from sampling import SampleAlgorithm
import PlyReader_fpfh
#import tensorflow as tf
import convnnutils
import time
import sys
import os.path
from scipy import spatial
#import fpfh


def map_section(Config, section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1



def get_scene(base_path, scenes_dir, curr_i, configs):
    #curr_i = (scene_i + 1) % scene_count
    #scene_i = curr_i
    
    config_path = os.path.join(scenes_dir, 'ConfigScene' + str(curr_i) + '.ini')
    if configs != None:
        config_path = configs[curr_i]
    
    
    print "config_path: ", config_path 
    config = ConfigParser.ConfigParser()
    #print "config path: ", config_path
    print "config read: ", config.read(config_path)
    print "config sections: ", config.sections()
    models_count = int(map_section(config, "MODELS")['number'].strip('\"'))
    model_path = [""]*models_count
    model_trans_path = [""]*models_count
    for i in range(models_count):
        model_path[i] = base_path + map_section(config, "MODELS")['model_' + str(i)].strip('\"')
        root, ext = os.path.splitext(model_path[i])
        model_path[i] = root + '_0' + ext
        model_trans_path[i] = base_path + map_section(config, "MODELS")['model_' + str(i) + '_groundtruth'].strip('\"')  
    scene_path = base_path + map_section(config, "SCENE")['path'].strip('\"')
    root, ext = os.path.splitext(scene_path)
    #scene_path = root + '_0.1' + ext
    scene_path = root + '_0' + ext
    
    return model_path, model_trans_path, scene_path

def fill_files_list(files_file, scenes_dir):
    files_list = []
    with open(files_file, 'r') as f:
        content = f.readlines()
    for file1 in content:
        f1 = os.path.join(scenes_dir, file1.rstrip('\n'))
        if os.path.isfile(f1):
            files_list.append(f1)
    return files_list
                

def filter_scene_samples(scene_data, scene_samples, scene_sample_indices, support_radii, threshold):
    tree = spatial.KDTree(scene_data)
    new_samples = []
    new_indices = []
    for sample_i, samplept in enumerate(scene_samples):
        i = tree.query_ball_point(samplept[0:3], r=support_radii)
        if len(i) > threshold:
            new_samples.append(samplept)
            new_indices.append(scene_sample_indices[sample_i])
    
    return numpy.asarray(new_samples), numpy.asarray(new_indices)

def get_indices(pc, samples):
    tree = spatial.KDTree(pc)
    indices = numpy.zeros((samples.shape[0],))
    for pt_i, samplept in enumerate(samples):
        _, index = tree.query(samplept, k=1)
        indices[pt_i] = index
    return indices


def main(args):

    scenes_dir = args[1]
    base_path = args[2]
    scenes_count = int(args[3])
    train_rotations = int(args[4])
    num_samples = int(args[5])
    ratio = float(args[6])
    rel_support_radii = float(args[7])
    patch_dim = 32
    relL = 0.05
    #config_files = fill_files_list(os.path.join(scenes_dir, "configs.txt"), scenes_dir)
    config_files = None
    fpfh_models = []
    fpfh_scenes = []
    for s in range(1, scenes_count):
        model_paths, model_trans_paths, scene_path = get_scene(base_path, scenes_dir, s, configs=config_files)
        #if "Scene3" in scene_path or "Scene4" in scene_path:
        #    continue
        print "scene path: ", scene_path
        reader_scene = PlyReader_fpfh.PlyReader()
        reader_scene.read_ply(scene_path, num_samples=num_samples, add_noise=False, noise_std=0.5, noise_prob=0, noise_factor=0,
                           rotation_axis=[0, 0, 1], rotation_angle=utils.rad(0), 
                           sampling_algorithm=SampleAlgorithm.ISS_Detector)
        pc_diameter = utils.get_pc_diameter(reader_scene.data)
        l_scene = relL*pc_diameter
        support_radii = rel_support_radii*pc_diameter
        #support_radii = 0.0114401621899
        print "supprot_radii", support_radii
        reader_scene.set_variables(l=l_scene, patch_dim=patch_dim, filter_bad_samples=False, filter_threshold=50, use_point_as_mean=False)
        print "num before filtering: ", reader_scene.samples.shape[0]
        reader_scene.samples, reader_scene.sample_indices = filter_scene_samples(reader_scene.data,
                                                                                  reader_scene.samples, reader_scene.sample_indices,
                                                                                   support_radii/2, threshold=500)
        print "num after filtering: ", reader_scene.samples.shape[0]
        reader_scene_samles_all = reader_scene.samples
        
        for model_num in range(len(model_paths)):
            model_path = model_paths[model_num]
            model_trans_path = model_trans_paths[model_num]
            #if not ("bun" in model_path):
            #    continue 
            print "trans mat path: ", model_trans_path
            print "model_path: ", model_path
            
            if not "chef" in model_path:
                print "skipping: ", model_path
                continue
            
                
            trans_mat = numpy.loadtxt(model_trans_path, ndmin=2)
            reader = PlyReader_fpfh.PlyReader()
            reader.read_ply(model_path, num_samples=num_samples, add_noise=False,
                             sampling_algorithm=SampleAlgorithm.ISS_Detector)
            pc_diameter = utils.get_pc_diameter(reader.data)
            l = relL*pc_diameter
            reader.set_variables(l=l, patch_dim=patch_dim, filter_bad_samples=False, filter_threshold=50, use_point_as_mean=False)
            
            ##for testing only
            #reader.set_variables(l=l_scene, patch_dim=patch_dim, filter_bad_samples=False, filter_threshold=50)
            reader_scene.set_variables(l=l, patch_dim=patch_dim, filter_bad_samples=False, filter_threshold=50, use_point_as_mean=False)
            
            """
            #root, ext = os.path.splitext(scene_path)
            if not (model_path in fpfh_models):
                root, ext = os.path.splitext(model_path)
                fpfh.compute_fpfh_all(root, l, l)
                fpfh_models.append(model_path)
            else:
                print "model exists: ", model_path
            
            
            if not (scene_path in fpfh_scenes):
                root, ext = os.path.splitext(scene_path)
                fpfh.compute_fpfh_all(root, l, l)
                fpfh_scenes.append(scene_path)
            else:
                print "scene exists: ", scene_path
            continue
            """   
            
            #reader_scene.samples = utils.transform_pc(reader.samples, trans_mat)
            reader_scene.sample_class_current = 0
            reader_scene.index = 0
            
            reader_scene.samples = utils.extrac_same_samples(reader.data, reader.samples, reader_scene.data, reader_scene.samples, trans_mat, support_radii)
            reader_scene.sample_indices = get_indices(reader_scene.data, reader_scene.samples)
            
            num_cf, _, _ = utils.num_corresponding_features(reader.samples, reader_scene_samles_all, reader_scene.data, trans_mat, support_radii)
            #reader.sample_indices = get_indices(reader.data, reader.samples)
            #reader_scene.sample_indices = get_indices(reader_scene.data, reader_scene.samples)
            print "num_corresponding_features: ", num_cf
            
            print s
            
            
            desc1 = reader.fpfh[numpy.array(reader.sample_indices, numpy.int32)]
            desc2 = reader_scene.fpfh[numpy.array(reader_scene.sample_indices, numpy.int32)]
            
            ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
            matches_arr = [None]*len(ratios)
            for ratio_i, ratio1 in enumerate(ratios):
                if desc1.shape[0] < desc2.shape[0]:                    
                    matches_arr[ratio_i] = utils.match_des_test(desc1, desc2, ratio1)
                    print "match_des_test"
                else:
                    matches_arr[ratio_i] = utils.match_des(desc1, desc2, ratio1)
                    print "match_des"
            
            #matches = utils.match_des(desc1, desc2, ratio)
            #print "match_des"
            #numpy.save("scene_debug/scene_matches.npy", matches)
            #print 'num_matches: ', len(matches)
            #correct, wrong = utils.correct_matches_support_radii(reader.samples, reader_scene.samples, matches,
            #                                       pose=trans_mat, N=100000, support_radii=support_radii)
            
            correct_arr = [None]*len(ratios)
            recall_arr = [None]*len(ratios)
            match_res_arr = [None]*len(ratios)
            
            for matches_i, matches in enumerate(matches_arr):
                correct_arr[matches_i], _, match_res_arr[matches_i] = utils.correct_matches_support_radii(reader_scene.samples, reader.samples, matches,
                                                       pose=trans_mat, N=100000, support_radii=support_radii)
                
            #correct, wrong = utils.correct_matches(reader.samples, reader_scene.samples, matches, N=100000)
            best10 = num_samples//10
            print 'N=', best10
            print 'total sample count', reader.samples.shape[0]
            correct10 = -1
            #correct10, wrong10 = utils.correct_matches_support_radii(reader.samples, reader_scene.samples, matches,
            #                                           pose=trans_mat, N=best10, support_radii=support_radii)
            
            for ratio_i, _ in enumerate(ratios):
                    recall_arr[ratio_i] = (len(matches_arr[ratio_i])/float(num_cf))*correct_arr[ratio_i]
            scene_name = os.path.split(scene_path)[1]
            for ratio_i, ratio1 in enumerate(ratios):
                with open("results_fpfh_mian_same{0}.txt".format(ratio1), "a") as myfile:
                    myfile.write('train rotations: ' + str(train_rotations) + '    num samples: ' + str(num_samples) + '    scene: ' + scene_name + "    correct: {0:.4f}".format(correct_arr[ratio_i]) + "    correct best 10: {0:.4f}".format(correct10) + "  after filtering count: " + str(reader.samples.shape[0]) + "  num matches: " + str(len(matches_arr[ratio_i]))  + " ratio : {0:.1f}".format(ratio1) + " recall final : {0:.4f}".format(recall_arr[ratio_i]) + '\n')
                    myfile.close()
                with open("precision_fpfh_mian_same{0}.txt".format(ratio1), "a") as myfile:
                    myfile.write("{0:.4f}".format(correct_arr[ratio_i]) + '\n')
                    myfile.close()
                with open("recall_fpfh_mian_same{0}.txt".format(ratio1), "a") as myfile:
                    myfile.write("{0:.4f}".format(recall_arr[ratio_i]) + '\n')
                    myfile.close()
            #plotutils.show_matches(reader.data, reader_noise.data, reader.samples, reader_noise.samples, matches, N=200)
        print 'done'

#if __name__ == "__main__":
#    main(sys.argv)
"""
main(["", r"/home/titan/hasan/workspace/Conv3d/retrieval_dataset/3D models/Stanford/RandomViews",
      "/home/titan/hasan/workspace/Conv3d/retrieval_dataset", 
      "36", "-1", "-1", "1", "0.07"])
"""

"""
main(["", r"/home/hasan/workspace/Conv3D/retrieval_dataset/3D models/Stanford/RandomViews",
      "/home/hasan/workspace/Conv3D/retrieval_dataset", 
      "36", "-1", "-1", "1", "0.07"])
"""

"""
main(["", r"/home/hasan/workspace/Conv3D/retrieval_dataset/3D models/Stanford/Retrieval",
      "/home/hasan/workspace/Conv3D/retrieval_dataset", 
      "18", "-1", "-1", "1", "0.07"])
"""
"""
main(["", r"/home/titan/hasan/workspace/Conv3d/retrieval_dataset/3D models/Stanford/Retrieval",
      "/home/titan/hasan/workspace/Conv3d/retrieval_dataset", 
      "18", "-1", "-1", "1", "0.07"])
"""

"""
main(["", r"/home/titan/hasan/workspace/Conv3d/laser/3D models/Mian",
      "/home/titan/hasan/workspace/Conv3d/laser", 
      "50", "-1", "-1", "1.1", "0.05"])
"""

"""
main(["", r"/home/hasan/workspace/Conv3D/laser/3D models/Mian",
      "/home/hasan/workspace/Conv3D/laser", 
      "51", "-1", "-1", "1", "0.07"])
"""

main(["", r"/home/hasan/hasan/workspace/Conv3d/laser/3D models/Mian",
      "/home/hasan/hasan/workspace/Conv3d/laser", 
      "51", "-1", "-1", "1", "0.07"])

