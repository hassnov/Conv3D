import ConfigParser
import os.path
import numpy
import utils
#import plotutils
#from mayavi import mlab
from plyfile import PlyData
from sampling import SampleAlgorithm
import PlyReader
import tensorflow as tf
import convnnutils
import time
import sys
import os.path
from scipy import spatial


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
                
def filter_scene_samples(scene_data, scene_samples, support_radii, threshold):
    tree = spatial.KDTree(scene_data)
    new_samples = []
    for samplept in scene_samples:
        i = tree.query_ball_point(samplept[0:3], r=support_radii)
        if len(i) > threshold:
            new_samples.append(samplept)
    
    return numpy.asarray(new_samples)

    
def main(args):
    models_dir = '/home/hasan/hasan/tr_models/'
    BATCH_SIZE=10
    num_rotations=1
    samples_per_batch = BATCH_SIZE * num_rotations
    
    
    #base_path = "/home/hasan/Downloads"
    scene_i = 0
    #scene_count = 51
    #scenes_dir = r"/home/hasan/Downloads/3D models/Stanford/Retrieval"
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
    
    for s in range(1, scenes_count):
        model_paths, model_trans_paths, scene_path = get_scene(base_path, scenes_dir, s, configs=config_files)
        #if "Scene3" in scene_path or "Scene4" in scene_path:
        #    continue
        print "scene path: ", scene_path
        reader_scene = PlyReader.PlyReader()
        reader_scene.read_ply(scene_path, num_samples=num_samples, add_noise=False, noise_std=0.5, noise_prob=0, noise_factor=0,
                           rotation_axis=[0, 0, 1], rotation_angle=utils.rad(0), 
                           sampling_algorithm=SampleAlgorithm.ISS_Detector)
        pc_diameter = utils.get_pc_diameter(reader_scene.data)
        l_scene = relL*pc_diameter
        support_radii = rel_support_radii*pc_diameter
        #support_radii = 0.0114401621899
        print "scene diameter: ", pc_diameter
        print "supprot_radii", support_radii
        reader_scene.set_variables(l=l_scene, patch_dim=patch_dim, filter_bad_samples=False, filter_threshold=50, use_point_as_mean=False)
        print "num before filtering: ", reader_scene.samples.shape[0]
        reader_scene.samples = filter_scene_samples(reader_scene.data, reader_scene.samples, support_radii/2, threshold=500)
        print "num after filtering: ", reader_scene.samples.shape[0]
        #reader_scene_samples_full = reader_scene.samples
        
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
            reader = PlyReader.PlyReader()
            reader.read_ply(model_path, num_samples=num_samples, add_noise=False,
                             sampling_algorithm=SampleAlgorithm.ISS_Detector)
            pc_diameter = utils.get_pc_diameter(reader.data)
            print "model diameter: ", pc_diameter
            l = relL*pc_diameter
            reader.set_variables(l=l, patch_dim=patch_dim, filter_bad_samples=False, filter_threshold=50, use_point_as_mean=False)
            
            #reader_scene.samples = utils.transform_pc(reader.samples, trans_mat)
            
            reader_scene.sample_indices = -1
            reader_scene.index = 0
            reader_scene.sample_class_current = 0
            ### just for testing
            print "l: {0}, l_scene: {1}".format(l, l_scene)
            reader_scene.set_variables(l=l, patch_dim=patch_dim, filter_bad_samples=False, filter_threshold=50, use_point_as_mean=False)
            
            reader_scene.samples = utils.extrac_same_samples(reader.data, reader.samples, reader_scene.data, reader_scene.samples, trans_mat, support_radii)
            
            num_cf, _, _ = utils.num_corresponding_features(reader.samples, reader_scene.samples, reader_scene.data, trans_mat, support_radii)
            #reader.samples = model_corres
            #reader_scene.samples = scene_corres
            print "num_corresponding_features: ", num_cf
            
            #numpy.save("scene_debug/scene_scene.npy", reader_scene.data)
            #numpy.save("scene_debug/scene_scene_samples.npy", reader_scene.samples)
            #numpy.save("scene_debug/scene_model_samples.npy", reader.samples)
            
            print "s: ", s
            #continue
            samples_count = reader.compute_total_samples(num_rotations)
            batches_per_epoch = samples_count/BATCH_SIZE
            
            with tf.Graph().as_default() as graph:
                #net_x = tf.placeholder("float", X.shape, name="in_x")
                #net_y = tf.placeholder(tf.int64, Y.shape, name="in_y")
                
                net_x = tf.placeholder("float", [samples_per_batch, patch_dim, patch_dim, patch_dim, 1], name="in_x")
                net_y = tf.placeholder(tf.int64, [samples_per_batch,], name="in_y")
                
                logits, regularizers, conv1, pool1, h_fc0, h_fc1 = convnnutils.build_graph_3d_5_5_3_3_3(net_x, 0.5, 2070, train=False)
                #logits, regularizers, conv1, pool1, h_fc0, h_fc1 = convnnutils.build_graph_3d_5_5_3_3_3(net_x, 0.5, 3057, train=False)
                #logits, regularizers, conv1, pool1, h_fc0, h_fc1 = convnnutils.build_graph_3d_5_5_3_3_3(net_x, 0.5, 5460, train=False)
                #logits, regularizers, conv1, pool1, h_fc0, h_fc1 = convnnutils.build_graph_3d_5_5_3_3_3_4000(net_x, 0.5, 5460, train=False)
                #logits, regularizers, conv1, pool1, h_fc0, h_fc1 = convnnutils.build_graph_3d_5_5_3_3_3_4000(net_x, 0.5, 113, train=False)
                #logits, regularizers, conv1, pool1, h_fc0, h_fc1 = convnnutils.build_graph_3d_5_5_3_3_small(net_x, 0.5, 537, train=False)
                
                #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, net_y))
                #loss += 5e-4 * regularizers
                
                print 'logits shape: ',logits.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
                print 'X shape: ',  net_x.get_shape().as_list()
                
                global_step = tf.Variable(0, trainable=False)
                
                correct_prediction = tf.equal(tf.argmax(logits,1), net_y)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
                # Create initialization "op" and run it with our session 
                init = tf.initialize_all_variables()
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.10)
                sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
                sess.run(init)
                
                # Create a saver and a summary op based on the tf-collection
                saver = tf.train.Saver(tf.all_variables())
                #saver.restore(sess, os.path.join(models_dir,'32models_'+ str(train_rotations) +'_5_5_3_3_3443_ae_kmeans_copy2.ckpt'))   # Load previously trained weights
                #if first:
                #saver.restore(sess, os.path.join(models_dir,'9models_270aug_5_5_3_3_3_7267_noise_NoClusters_copy.ckpt'))
                
                saver.restore(sess, os.path.join(models_dir,'6models_720aug_5_5_3_3_3_2070_90rots_noise_relR_nr_200clusters_fpfh_wd2.ckpt'))
                #saver.restore(sess, os.path.join(models_dir,'5models_720aug_5_5_3_3_3_3057_90rots_noise_relR_nr_400clusters_fpfh.ckpt'))
                #saver.restore(sess, os.path.join(models_dir,'6models_720aug_5_5_3_3_3_2070_90rots_noise_relR_nr_NoClusters.ckpt'))
                #saver.restore(sess, os.path.join(models_dir,'5models_720aug_5_5_3_3_3_3057_noise_nr_relR_NoClusters_copy.ckpt'))
                #saver.restore(sess, os.path.join(models_dir,'9models_540aug_5_5_3_3_3_5460_45rots_noise_relR_nr_NoClusters.ckpt'))
                #saver.restore(sess, os.path.join(models_dir,'1models_540aug_5_5_3_3_537_arm_45rots_noise__relR_nr_NoClusters.ckpt'))
                
                #saver.restore(sess, os.path.join(models_dir,'9models_540aug_5_5_3_3_3_5460_45rots_noise_relR_nr_800Clusters_copy.ckpt'))
                #saver.restore(sess, os.path.join(models_dir,'9models_540aug_5_5_3_3_3_5460_90rots_relR_nr_NoClusters_copy.ckpt'))
                
                #saver.restore(sess, os.path.join(models_dir,'5models_360aug_5_5_3_3_3_3057_noise_nr_NoClusters_copy0.66.ckpt'))
                #saver.restore(sess, os.path.join(models_dir,'45models_360aug_5_5_3_3_3_10000_nr_noise_NoClusters.ckpt'))  
                #saver.restore(sess, os.path.join(models_dir,'5models_360aug_5_5_3_3057_noise_nr_NoClusters_copy.ckpt'))
                
                first = False
                print [v.name for v in tf.all_variables()]
                b = 0
                
                c1_shape = conv1.get_shape().as_list()
                p1_shape = pool1.get_shape().as_list()
                f0_shape = h_fc0.get_shape().as_list()
                f1_shape = h_fc1.get_shape().as_list()
                samples_count_mod_patch = reader.samples.shape[0] - reader.samples.shape[0] % BATCH_SIZE
                c1_1s = numpy.zeros((reader.samples.shape[0], c1_shape[1] * c1_shape[2] * c1_shape[3] * c1_shape[4]), dtype=numpy.float32)
                p1_1s = numpy.zeros((reader.samples.shape[0], p1_shape[1] * p1_shape[2] * p1_shape[3] * p1_shape[4]), dtype=numpy.float32)
                f0_1s = numpy.zeros((samples_count_mod_patch, f0_shape[1]*3), dtype=numpy.float32)
                f1_1s = numpy.zeros((samples_count_mod_patch, f1_shape[1]), dtype=numpy.float32)
                
                samples_count_scene_mod_patch = reader_scene.samples.shape[0] - reader_scene.samples.shape[0] % BATCH_SIZE
                c1_2s = numpy.zeros((reader.samples.shape[0], c1_shape[1] * c1_shape[2] * c1_shape[3] * c1_shape[4]), dtype=numpy.float32)
                p1_2s = numpy.zeros((reader.samples.shape[0], p1_shape[1] * p1_shape[2] * p1_shape[3] * p1_shape[4]), dtype=numpy.float32)
                f0_2s = numpy.zeros((samples_count_scene_mod_patch, f0_shape[1]*3), dtype=numpy.float32)
                f1_2s = numpy.zeros((samples_count_scene_mod_patch, f1_shape[1]), dtype=numpy.float32)
                
                for b in range(reader_scene.samples.shape[0] // BATCH_SIZE):
                    i = b*num_rotations*BATCH_SIZE
                    i1 = (b + 1)*num_rotations*BATCH_SIZE 
                    X2, Y2= reader_scene.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations, increment=False)
                    X207, Y207 = reader_scene.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations, increment=False, r_in=reader_scene.l*(0.04/relL))
                    X203, Y203 = reader_scene.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations, increment=True, r_in=reader_scene.l*(0.03/relL))
                    #X202, Y202 = reader_scene.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations, increment=True, r_in=reader_scene.l*(0.02/relL))
                    """
                    numpy.save('scene_debug/sample_scene_X_' + str(b), X2)
                    numpy.save('scene_debug/sample_scene_X02_' + str(b), X202)
                    numpy.save('scene_debug/sample_scene_X03_' + str(b), X203)
                    numpy.save('scene_debug/sample_scene_X05_' + str(b), X207)
                    """
                    _, _, f0_2, f1_2 = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X2, net_y: Y2})
                    _, _, f0_207, _ = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X207, net_y: Y207})
                    _, _, f0_203, _ = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X203, net_y: Y203})
                    #_, _, f0_202, _ = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X202, net_y: Y202})
                    #assert (numpy.all(f0_2 == f0_207))
                    #print b, ", ",
                    #f0_2 = numpy.hstack((f0_2, f0_207, f0_203, f0_202))                    
                    f0_2 = numpy.hstack((f0_2, f0_207, f0_203))
                    f0_2s[i:i1] = numpy.reshape(f0_2, (samples_per_batch, f0_2s.shape[1]))
                    f1_2s[i:i1] = numpy.reshape(f1_2, (samples_per_batch, f1_2s.shape[1]))
                    #print b

                for b in range(reader.samples.shape[0] // BATCH_SIZE):
                    start_time = time.time()
                    X, Y= reader.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations, increment=False)
                    X07, Y07 = reader.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations, increment=False, r_in=reader.l*(0.04/relL))
                    X03, Y03 = reader.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations, increment=True, r_in=reader.l*(0.03/relL))
                    #X02, Y02 = reader.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations, increment=True, r_in=reader.l*(0.02/relL))
                    patch_time = time.time() - start_time
                    """                 
                    numpy.save('scene_debug/sample_model_X_' + str(b), X)
                    numpy.save('scene_debug/sample_model_X02_' + str(b), X02)
                    numpy.save('scene_debug/sample_model_X03_' + str(b), X03)
                    numpy.save('scene_debug/sample_model_X05_' + str(b), X07)
                    """
                    i = b*num_rotations*BATCH_SIZE
                    i1 = (b + 1)*num_rotations*BATCH_SIZE
                    start_eval = time.time()
                    _, _, f0_1, f1_1 = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X, net_y: Y})
                    _, _, f0_107, _ = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X07, net_y: Y07})
                    _, _, f0_103, _ = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X03, net_y: Y03})
                    #_, _, f0_102, _ = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X02, net_y: Y02})
                    eval_time = time.time() - start_eval 
                    #assert (numpy.all(f0_1 == f0_107))                  
                    #f0_1 = numpy.hstack((f0_1, f0_107, f0_103, f0_102))
                    f0_1 = numpy.hstack((f0_1, f0_107, f0_103))
                    f0_1s[i:i1] = numpy.reshape(f0_1, (samples_per_batch, f0_1s.shape[1]))
                    f1_1s[i:i1] = numpy.reshape(f1_1, (samples_per_batch, f1_1s.shape[1]))                   
                    duration = time.time() - start_time
                    print "point:", b, "  patch time: {0:.2f}".format(patch_time) ,"    eval time: {0:.2f}".format(eval_time), "   Duration (sec): {0:.2f}".format(duration)#, "    loss:    ", error, "    Accuracy: ", acc #, "   Duration (sec): ", duration
                    
                    
                print 'total'
                
                #numpy.save('scene_debug/scene_f0_1s.npy', f0_1s)
                #numpy.save('scene_debug/scene_f0_2s.npy', f0_2s)
                desc1 = f0_1s
                desc2 = f0_2s
                
                
                ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
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
                #numpy.save('scene_debug/matches', matches)
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
                #numpy.save('scene_debug/matche_res', match_res)
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
                    with open("results_mian_6models_same{0}.txt".format(ratio1), "a") as myfile:
                        myfile.write('train rotations: ' + str(train_rotations) + '    num samples: ' + str(num_samples) + '    scene: ' + scene_name + "    correct: {0:.4f}".format(correct_arr[ratio_i]) + "    correct best 10: {0:.4f}".format(correct10) + "  after filtering count: " + str(reader.samples.shape[0]) + "  num matches: " + str(len(matches_arr[ratio_i]))  + " ratio : {0:.1f}".format(ratio1) + " recall final : {0:.4f}".format(recall_arr[ratio_i]) + '\n')
                        myfile.close()
                    with open("precision_mian_6models_same{0}.txt".format(ratio1), "a") as myfile:
                        myfile.write("{0:.4f}".format(correct_arr[ratio_i]) + '\n')
                        myfile.close()
                    with open("recall_mian_6models_same{0}.txt".format(ratio1), "a") as myfile:
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
main(["", r"/home/titan/hasan/workspace/Conv3d/retrieval_dataset/3D models/Stanford/Retrieval",
      "/home/titan/hasan/workspace/Conv3d/retrieval_dataset", 
      "18", "-1", "-1", "1", "0.07"])
"""


main(["", r"/home/hasan/hasan/workspace/Conv3d/laser/3D models/Mian",
      "/home/hasan/hasan/workspace/Conv3d/laser", 
      "51", "-1", "-1", "1", "0.07"])


"""
main(["", r"/home/hasan/Downloads/UWA/3D models/Mian",
      "/home/hasan/Downloads/UWA", 
      "51", "40", "100", "1.1", "0.04"])
"""
