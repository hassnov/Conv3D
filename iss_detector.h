#ifndef ISS_DETECTOR_H
#define ISS_DETECTOR_H
#include "header.h"
#include <pcl/keypoints/iss_3d.h>
#endif // ISS_DETECTOR_H

pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detect(pcl::PointCloud<pcl::PointXYZ>::Ptr model,
                                               pcl::PointCloud<pcl::PointXYZ>::Ptr & model_keypoints,
                                               pcl::IndicesConstPtr & indices);

double computeCloudResolution (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud);
