#import scene_test
#import scene_test_real
import scene_test_fpfh

#ratios = [10, 0.8, 0.6, 0.4, 0.2]
ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
ratios = [0.9, 0.8, 0.7, 0.6, 0.5]
ratios = [0.4, 0.3, 0.2]
for ratio in ratios:
    #scene_test_real.main(["", r"/home/titan/hasan/workspace/Conv3d/retrieval_dataset/3D models/Stanford/Retrieval",
    #      "/home/titan/hasan/workspace/Conv3d/retrieval_dataset", 
    #      "18", "-1", "-1", str(ratio), "0.07"])
    print "calling for ratio: ", ratio
    scene_test_fpfh.main(["", r"/home/hasan/workspace/Conv3D/retrieval_dataset/3D models/Stanford/RandomViews",
      "/home/hasan/workspace/Conv3D/retrieval_dataset", 
      "36", "-1", "-1", str(ratio), "0.07"])