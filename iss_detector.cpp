#include "iss_detector.h"
//#include "header.h"

double computeCloudResolution (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<pcl::PointXYZ> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}


pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detect(pcl::PointCloud<pcl::PointXYZ>::Ptr model,
                                               pcl::PointCloud<pcl::PointXYZ>::Ptr &model_keypoints,
                                               pcl::IndicesConstPtr &indices)
{
    double iss_salient_radius_;
    double iss_non_max_radius_;
    double iss_gamma_21_ (0.975);
    double iss_gamma_32_ (0.975);
    double iss_min_neighbors_ (5);
    int iss_threads_ (4);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr
    //model_keypoints = new pcl::PointCloud<pcl::PointXYZ>;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    double model_resolution = computeCloudResolution(model);
    iss_salient_radius_ = 6 * model_resolution;
    iss_non_max_radius_ = 4 * model_resolution;

    //
    // Compute keypoints
    //
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;

    iss_detector.setSearchMethod (tree);
    iss_detector.setSalientRadius (iss_salient_radius_);
    iss_detector.setNonMaxRadius (iss_non_max_radius_);
    iss_detector.setThreshold21 (iss_gamma_21_);
    iss_detector.setThreshold32 (iss_gamma_32_);
    iss_detector.setMinNeighbors (iss_min_neighbors_);
    iss_detector.setNumberOfThreads (iss_threads_);
    iss_detector.setInputCloud (model);
    iss_detector.compute (*model_keypoints);
    indices = iss_detector.getIndices();

    return iss_detector;
}
