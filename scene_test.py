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



def get_scene(base_path, scenes_dir, curr_i):
    #curr_i = (scene_i + 1) % scene_count
    #scene_i = curr_i
     
    config_path = os.path.join(scenes_dir, 'ConfigScene' + str(curr_i) + '.ini') 
    config = ConfigParser.ConfigParser()
    print "config path: ", config_path
    print "config read: ", config.read(config_path)
    print "config sections: ", config.sections()
    model_path = base_path + map_section(config, "MODELS")['model_0'].strip('\"')
    model_trans_path = base_path + map_section(config, "MODELS")['model_0_groundtruth'].strip('\"')  
    scene_path = base_path + map_section(config, "SCENE")['path'].strip('\"')
    root, ext = os.path.splitext(scene_path)
    scene_path = root + '_0.1' + ext
    
    return model_path, model_trans_path, scene_path

def main(args):
    models_dir = '/home/titan/hasan/tr_models/'
    BATCH_SIZE=10
    num_rotations=1
    samples_per_batch = BATCH_SIZE * num_rotations
    
    
    #base_path = "/home/hasan/Downloads"
    scene_i = 0
    scene_count = 16
    #scenes_dir = r"/home/hasan/Downloads/3D models/Stanford/Retrieval"
    scenes_dir = args[1]
    base_path = args[2]
    scenes_count = int(args[3])
    train_rotations = int(args[4])
    num_samples = int(args[5])
    ratio = float(args[6])
    support_radii = float(args[7])
    patch_dim = 32
    relL = 0.07
    for s in range(0, scenes_count):
        model_path, model_trans_path, scene_path = get_scene(base_path, scenes_dir, s)
        
        trans_mat = numpy.loadtxt(model_trans_path, delimiter=' ', ndmin=2)
        
        reader = PlyReader.PlyReader()
        reader.read_ply(model_path, num_samples=num_samples, add_noise=False, sampling_algorithm=SampleAlgorithm.ISS_Detector)
        pc_diameter = utils.get_pc_diameter(reader.data)
        l = relL*pc_diameter
        reader.set_variables(l=l, patch_dim=patch_dim, filter_bad_samples=False, filter_threshold=50)
        
        reader_scene = PlyReader.PlyReader()
        reader_scene.read_ply(scene_path, num_samples=num_samples, add_noise=False, sampling_algorithm=SampleAlgorithm.ISS_Detector)
        pc_diameter = utils.get_pc_diameter(reader_scene.data)
        l_scene = relL*pc_diameter
        support_radii = support_radii*pc_diameter
        print "supprot_radii", support_radii
        reader_scene.set_variables(l=l_scene, patch_dim=patch_dim, filter_bad_samples=False, filter_threshold=50)
        reader_scene.samples = utils.transform_pc(reader.samples, trans_mat)
        reader_scene.sample_indices = -1
        
        samples_count = reader.compute_total_samples(num_rotations)
        batches_per_epoch = samples_count/BATCH_SIZE
        
        with tf.Graph().as_default() as graph:
            #net_x = tf.placeholder("float", X.shape, name="in_x")
            #net_y = tf.placeholder(tf.int64, Y.shape, name="in_y")
            
            net_x = tf.placeholder("float", [samples_per_batch, patch_dim, patch_dim, patch_dim, 1], name="in_x")
            net_y = tf.placeholder(tf.int64, [samples_per_batch,], name="in_y")
            
            logits, regularizers, conv1, pool1, h_fc0, h_fc1 = convnnutils.build_graph_3d_5_3_nopool(net_x, 0.5, 500, train=False)
            
            #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, net_y))
            #loss += 5e-4 * regularizers
            
            print 'logits shape: ',logits.get_shape().as_list(), ' net_y shape: ', net_y.get_shape().as_list()
            print 'X shape: ',  net_x.get_shape().as_list()
            
            global_step = tf.Variable(0, trainable=False)
            
            correct_prediction = tf.equal(tf.argmax(logits,1), net_y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
            # Create initialization "op" and run it with our session 
            init = tf.initialize_all_variables()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options))
            sess.run(init)
            
            # Create a saver and a summary op based on the tf-collection
            saver = tf.train.Saver(tf.all_variables())
            saver.restore(sess, os.path.join(models_dir,'bunny_'+ str(train_rotations) +'_5_3_500_filter40.ckpt'))   # Load previously trained weights
            
            print [v.name for v in tf.all_variables()]
            b = 0
            
            c1_shape = conv1.get_shape().as_list()
            p1_shape = pool1.get_shape().as_list()
            f0_shape = h_fc0.get_shape().as_list()
            f1_shape = h_fc1.get_shape().as_list()
            c1_1s = numpy.zeros((reader.samples.shape[0], c1_shape[1] * c1_shape[2] * c1_shape[3] * c1_shape[4]), dtype=numpy.float32)
            p1_1s = numpy.zeros((reader.samples.shape[0], p1_shape[1] * p1_shape[2] * p1_shape[3] * p1_shape[4]), dtype=numpy.float32)
            f0_1s = numpy.zeros((reader.samples.shape[0], f0_shape[1]), dtype=numpy.float32)
            f1_1s = numpy.zeros((reader.samples.shape[0], f1_shape[1]), dtype=numpy.float32)
            
            c1_2s = numpy.zeros((reader.samples.shape[0], c1_shape[1] * c1_shape[2] * c1_shape[3] * c1_shape[4]), dtype=numpy.float32)
            p1_2s = numpy.zeros((reader.samples.shape[0], p1_shape[1] * p1_shape[2] * p1_shape[3] * p1_shape[4]), dtype=numpy.float32)
            f0_2s = numpy.zeros((reader.samples.shape[0], f0_shape[1]), dtype=numpy.float32)
            f1_2s = numpy.zeros((reader.samples.shape[0], f1_shape[1]), dtype=numpy.float32)
            
            #for b in range(100):
            for b in range(samples_count // BATCH_SIZE):
                start_time = time.time()
                X, Y= reader.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations)
                patch_time = time.time() - start_time
                X2, Y2= reader_scene.next_batch_3d(BATCH_SIZE, num_rotations=num_rotations)
                
                i = b*num_rotations*BATCH_SIZE
                i1 = (b + 1)*num_rotations*BATCH_SIZE
                start_eval = time.time()
                c1_1, p1_1, f0_1, f1_1 = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X, net_y: Y})
                eval_time = time.time() - start_eval
                c1_1s[i:i1] = numpy.reshape(c1_1, (samples_per_batch,c1_1s.shape[1]))
                p1_1s[i:i1] = numpy.reshape(p1_1, (samples_per_batch, p1_1s.shape[1]))
                f0_1s[i:i1] = numpy.reshape(f0_1, (samples_per_batch, f0_1s.shape[1]))
                f1_1s[i:i1] = numpy.reshape(f1_1, (samples_per_batch, f1_1s.shape[1]))
                
                c1_2, p1_2, f0_2, f1_2 = sess.run([conv1, pool1, h_fc0, h_fc1], feed_dict={net_x:X2, net_y: Y2})
                c1_2s[i:i1] = numpy.reshape(c1_2, (samples_per_batch, c1_2s.shape[1]))
                p1_2s[i:i1] = numpy.reshape(p1_2, (samples_per_batch, p1_2s.shape[1]))
                f0_2s[i:i1] = numpy.reshape(f0_2, (samples_per_batch, f0_2s.shape[1]))
                f1_2s[i:i1] = numpy.reshape(f1_2, (samples_per_batch, f1_2s.shape[1]))
                duration = time.time() - start_time
                if b % 10 == 0:
                    matches = utils.match_des(f1_1s[i:i1], f1_2s[i:i1], ratio)
                    utils.correct_matches_support_radii(reader.samples[i:i1], reader_scene.samples[i:i1],
                                                         matches, pose=trans_mat, N=samples_count,
                                                          support_radii=support_radii)
                
                
            print "point:", b, "  patch time: {0:.2f}".format(patch_time) ,"    eval time: {0:.2f}".format(eval_time), "   Duration (sec): {0:.2f}".format(duration)#, "    loss:    ", error, "    Accuracy: ", acc #, "   Duration (sec): ", duration
            print 'total'
            matches = utils.match_des(f1_1s, f1_2s, ratio)
            print 'num_matches: ', len(matches)
            correct, wrong = utils.correct_matches_support_radii(reader.samples, reader_scene.samples, matches,
                                                   pose=trans_mat, N=100000, support_radii=support_radii)
            best10 = num_samples//10
            print 'N=', best10
            print 'total sample count', reader.samples.shape[0]
            correct10, wrong10 = utils.correct_matches_support_radii(reader.samples, reader_scene.samples, matches,
                                                       pose=trans_mat, N=best10, support_radii=support_radii)
            recall = (len(matches)/float(reader.samples.shape[0])*correct)
            scene_name = os.path.split(scene_path)[1]
            with open("results_scene.txt", "a") as myfile:
                myfile.write('train rotations: ' + str(train_rotations) + '    num samples: ' + str(num_samples) + '    scene: ' + scene_name + "    correct: {0:.4f}".format(correct) + "    correct best 10: {0:.4f}".format(correct10) + "  after filtering count: " + str(reader.samples.shape[0]) + "  num matches: " + str(len(matches))  + " ratio : {0:.1f}".format(ratio) + " recall final : {0:.4f}".format(recall) + '\n')
                myfile.close()
            with open("precision.txt", "a") as myfile:
                myfile.write("{0:.4f}".format(correct) + '\n')
                myfile.close()
            with open("recall.txt", "a") as myfile:
                myfile.write("{0:.4f}".format(recall) + '\n')
                myfile.close()
            #plotutils.show_matches(reader.data, reader_noise.data, reader.samples, reader_noise.samples, matches, N=200)
        print 'done'
    
#if __name__ == "__main__":
#    main(sys.argv)


main(["", r"/home/titan/hasan/workspace/Conv3d/retrieval_dataset/3D models/Stanford/Retrieval",
      "/home/titan/hasan/workspace/Conv3d/retrieval_dataset", 
      "18", "40", "100", "1.1", "0.04"])

