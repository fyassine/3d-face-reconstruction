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
private:
    static std::vector<Vector3d> searchForLandmarks(std::vector<double> depthValues, const Matrix3d& intrinsics, const Matrix4d& extrinsics);
    static void convertVideoFrameToPng(rs2::video_frame videoFrame);
    static Vector3d convert2Dto3D(const Eigen::Vector2d& point, double depth, const Eigen::Matrix3d& intrinsics, const Eigen::Matrix4d& extrinsics);
};


#endif //FACE_RECONSTRUCTION_INPUTDATAEXTRACTOR_H
