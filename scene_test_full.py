#import scene_test
#import scene_test_real
#import scene_test_real2

#import scene_teststan_opt
#import scene_teststan_optsame
import scene_testret_optsame
#import scene_testret_opt

#ratios = [10, 0.8, 0.6, 0.4, 0.2]
ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
#ratios = [0.9, 0.8, 0.7, 0.6, 0.5]
#ratios = [0.4, 0.3, 0.2]




trained_models = ['1models_720aug_5_5_3_3_3_537_full.ckpt']
prefix = ['arm_full']
num_clusters = [537]

trained_models1 = ['5models_720aug_5_5_3_3_3_3057_noise_nr_relR_NoClusters_copy.ckpt']
prefix1 = ['NoClustersWithNoise0.5']
num_clusters1 = [3057]

trained_models1 = ['1models_360aug_5_5_3_3_3_537_90rots_noise_relR.ckpt']
prefix1 = ['1Nr']
num_clusters1 = [537]


trained_models1 = ['1models_360aug_5_5_3_3_3_537_90rots_noise_nr.ckpt',
                  '1models_360aug_5_5_3_3_3_537_90rots_noise_relR.ckpt']
prefix1 = ['1Radii', '1Nr']
num_clusters1 = [537, 537]

trained_models1 = ['1models_8aug_5_5_3_3_3_537_1rots_noise_relR_nr.ckpt',
                  '1models_360aug_5_5_3_3_3_537_90rots_relR_nr.ckpt']
prefix1 = ['1Rots', 'NoNoise']
num_clusters1 = [537, 537]


trained_models1 = ['5models_720aug_5_5_3_3_3_3057_90rots_noise_relR_nr_108clusters_ae.ckpt']
prefix1 = ['ae109']
num_clusters1 = [109]

trained_models1 = ['5models_720aug_5_5_3_3_3_3057_90rots_noise_relR_nr_100clusters_fpfh.ckpt',
                  '5models_720aug_5_5_3_3_3_3057_90rots_noise_relR_nr_200clusters_fpfh.ckpt',
                  '5models_720aug_5_5_3_3_3_3057_90rots_noise_relR_nr_400clusters_fpfh.ckpt',
                  '5models_720aug_5_5_3_3_3_3057_90rots_noise_relR_nr_200clusters_eigen.ckpt',
                  '5models_720aug_5_5_3_3_3_3057_noise_nr_relR_NoClusters_copy.ckpt']

prefix1 = ['fpfh100', 'fpfh200', 'fpfh400', 'eigen200', 'NoClusters']
num_clusters1 = [100, 200, 400, 200, 3057]


for model_i in range(len(trained_models)):
    print "###################### MODEL {0} ######################".format(trained_models[model_i])
    
    scene_testret_optsame.main(["", r"/home/hasan/hasan/workspace/Conv3d/retrieval_dataset/3D models/Stanford/Retrieval",
                            "/home/hasan/hasan/workspace/Conv3d/retrieval_dataset", 
                            "18", "-1", "-1", "1", "0.07", trained_models[model_i], str(num_clusters[model_i]), prefix[model_i]])
    
    
    """
    scene_teststan_opt.main(["", r"/home/hasan/hasan/workspace/Conv3d/retrieval_dataset/3D models/Stanford/RandomViews",
      "/home/hasan/hasan/workspace/Conv3d/retrieval_dataset", 
      "36", "-1", "-1", "1", "0.07", trained_models[model_i], str(num_clusters[model_i]), prefix[model_i]])
    """
    
"""
    scene_testret_optsame.main(["", r"/home/hasan/hasan/workspace/Conv3d/retrieval_dataset/3D models/Stanford/Retrieval",
      "/home/hasan/hasan/workspace/Conv3d/retrieval_dataset", 
      "18", "-1", "-1", "1", "0.07", trained_models[model_i], str(num_clusters[model_i]), prefix[model_i]])

"""

    

