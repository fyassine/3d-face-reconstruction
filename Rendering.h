#ifndef FACE_RECONSTRUCTION_RENDERING_H
#define FACE_RECONSTRUCTION_RENDERING_H

#include <dlib/opencv.h>
//#include "opencv2/imgcodecs.hpp"
#include "BFMParameters.h"

/*void renderImage(cv::Mat inputImage){
    cv::Mat renderedImage;
    cv::imwrite("result.png", renderedImage);
}*/

static BfmProperties getProperties(const std::string& path){
    BfmProperties properties;
    initializeBFM(path, properties);
    return properties;
}

static void convertParametersToPly(const BfmProperties& properties, const std::string& resultPath){

    std::ofstream outFile(resultPath);
    //Header
    outFile << "ply" << std::endl;
    outFile << "format ascii 1.0" << std::endl;
    outFile << "element vertex " << properties.numberOfVertices << std::endl;
    outFile << "property float x" << std::endl;
    outFile << "property float y" << std::endl;
    outFile << "property float z" << std::endl;
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
    outFile << "property uchar alpha" << std::endl;
    outFile << "element face " << properties.numberOfTriangles << std::endl;
    outFile << "property list uchar int vertex_indices" << std::endl;
    outFile << "end_header" << std::endl;
    //Vertices
    for (int i = 0; i < properties.numberOfVertices * 3; i+=3) {
        //Position
        auto x = properties.shapeMean[i] + properties.expressionMean[i];
        auto y = properties.shapeMean[i + 1] + properties.expressionMean[i + 1];
        auto z = properties.shapeMean[i + 2] + properties.expressionMean[i + 2];
        //Color
        auto r = (int) (properties.colorMean[i] * 255);
        auto g = (int) (properties.colorMean[i + 1] * 255);
        auto b = (int) (properties.colorMean[i + 2] * 255);
        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }
    //Faces
    for (int i = 0; i < properties.numberOfTriangles * 3; i+=3) {
        outFile << "3 " << properties.triangles[i] << " " << properties.triangles[i + 1] << " " << properties.triangles[i + 2] << std::endl;
    }
    outFile.close();
}

/*#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "Eigen.h"
#include "hdf5.h" //TODO: Only for testing, remove later!
*/
/**
 * Render BFM on image
 * @param bfmVertices       3D vertices of the BFM, //Note: bfm_vertices mean?! because if so, then we can simply add the difference of shape and expression
 * @param
 *
 * */
 /*
void renderImage(const std::vector<Eigen::Vector3f>& bfmVertices,
                 const std::vector<std::vector<int>>& triangles,
                 const cv::Mat& rvec,
                 const cv::Mat& tvec,
                 const cv::Mat& intrinsics,
                 const cv::Mat& distCoeffs,
                 cv::Mat& image){
    //conversion to opencv coordinates: (because somehow Eigen::Vector3f doesn't work)
    std::vector<cv::Point3f> convertedVertices;
    for (const auto& vertex : bfmVertices){
        convertedVertices.emplace_back(vertex[0], vertex[1], vertex[2]);
    }

    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(convertedVertices, rvec, tvec, intrinsics, distCoeffs, projected_points);

    for (const auto& triangle : triangles) {
        cv::line(image, projected_points[triangle[0]], projected_points[triangle[1]], cv::Scalar(255, 0, 0), 1);
        cv::line(image, projected_points[triangle[1]], projected_points[triangle[2]], cv::Scalar(255, 0, 0), 1);
        cv::line(image, projected_points[triangle[2]], projected_points[triangle[0]], cv::Scalar(255, 0, 0), 1);
    }
}

void testRendering(const std::string pathToImage,
                   const std::string pathToBFM,
                   const cv::Mat intrinsics,
                   const std::vector<Eigen::Vector3f>& landmarks3D,
                   const std::vector<Eigen::Vector2f>& landmarks2D){

    std::vector<cv::Point3f> convertedLandmarks3D;
    for (const auto& vertex : landmarks3D){
        convertedLandmarks3D.emplace_back(vertex[0], vertex[1], vertex[2]);
    }

    std::vector<cv::Point2f> convertedLandmarks2D;
    for (const auto& vertex : landmarks2D){
        convertedLandmarks2D.emplace_back(vertex[0], vertex[1]);
    }

    cv::Mat image = cv::imread(pathToImage);

    std::vector<Eigen::Vector3f> bfmVertices;
    std::vector<std::vector<int>> triangles;

    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat rvec, tvec;
    cv::solvePnP(convertedLandmarks3D, convertedLandmarks2D, intrinsics, distCoeffs, rvec, tvec);
    renderImage(bfmVertices, triangles, rvec, tvec, intrinsics, distCoeffs, image);

    cv::imshow("Rendering", image);
    cv::waitKey(0);
}
*/
#endif //FACE_RECONSTRUCTION_RENDERING_H
