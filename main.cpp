//#include "header.h"
#include "iss_detector.h"


using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

pcl::PointCloud<pcl::Normal>::Ptr compute_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double radiusSearch = 0.03)
{
    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (radiusSearch);

    // Compute the features
    ne.compute (*cloud_normals);
    return cloud_normals;

    // cloud_normals->points.size () should have the same size as the input cloud->points.size ()*
}


pcl::PointCloud<pcl::FPFHSignature33>::Ptr myFPFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    normals = compute_normals(cloud);

    // Create the FPFH estimation class, and pass the input dataset+normals to it
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud (cloud);
    fpfh.setInputNormals (normals);
    // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

    // Create an empty kdtree representation, and pass it to the FPFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

    fpfh.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    fpfh.setRadiusSearch (0.05);

    // Compute the features
    fpfh.compute (*fpfhs);
    return fpfhs;
}


int main (int argc, char** argv)
{

    std::vector<int> p_file_indices;
    p_file_indices = parse_file_extension_argument (argc, argv, ".ply");
    if (p_file_indices.size () != 2)
    {
        print_error ("Need one input ply file and one output ply file to continue.\n");
        return (-1);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PLYReader reader;
    //reader.read ("/home/hasan/qtcreator_projects/pcl_test_qt/bunny.ply", *cloud);
    reader.read (argv[p_file_indices[0]], *cloud);
    std::cerr << "PointCloud: " << cloud->width * cloud->height
       << " data points (" << pcl::getFieldsList (*cloud) << ").";

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints (new pcl::PointCloud<pcl::PointXYZ>);
    //pcl::IndicesPtr indices (new std::vector <int>);
    pcl::IndicesConstPtr indices (new std::vector <int>);
    //pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints (iss_detect(cloud));

    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss (iss_detect(cloud, model_keypoints, indices));
    std::cout << std::endl << "detect_done: " << model_keypoints->width * model_keypoints->height
             << " data points (" << pcl::getFieldsList (*model_keypoints) << ")." << std::endl;
    //std::cout<< "indices: " << model_keypoints->size()<< std::endl;
    pcl::PLYWriter writer;
    writer.write(argv[p_file_indices[1]], *model_keypoints, false);
    //writer.writeASCII(argv[p_file_indices[1]], *model_keypoints);

    return 1;
}

