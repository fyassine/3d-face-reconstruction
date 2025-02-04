#ifndef FACE_RECONSTRUCTION_INPUTDATAEXTRACTOR_H
#define FACE_RECONSTRUCTION_INPUTDATAEXTRACTOR_H

#include <string>
#include "InputData.h"
#include <librealsense2/rs.hpp>
#include "FacialLandmarks.h"

class InputDataExtractor {
public:
    InputDataExtractor();
    ~InputDataExtractor();
    static InputData extractInputData(const std::string& path);
    static Vector3d convert2Dto3D(const Eigen::Vector2d& point, double depth, const Eigen::Matrix3d& intrinsics, const Eigen::Matrix4d& extrinsics);
    static Vector2d convert3Dto2D(const Eigen::Vector3d& point, const Eigen::Matrix3d& depthIntrinsics, const Eigen::Matrix4d& extrinsics);
private:
    static std::vector<Vector3d> searchForLandmarks(std::vector<double> depthValues, const Matrix3d& intrinsics, const Matrix4d& extrinsics);
    static void convertVideoFrameToPng(rs2::video_frame videoFrame);
};


#endif //FACE_RECONSTRUCTION_INPUTDATAEXTRACTOR_H
