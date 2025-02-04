#ifndef FACE_RECONSTRUCTION_SINGLEINPUTFRAME_H
#define FACE_RECONSTRUCTION_SINGLEINPUTFRAME_H

#include "Eigen.h"

class SingleInputFrame {
public:
    SingleInputFrame(std::vector<Vector3d> rgbData, std::vector<double> depthData, std::vector<Vector3d> landmarks);
    ~SingleInputFrame();
    const std::vector<Vector3d> &getMLandmarks() const;

    const std::vector<Vector3d> &getMRgbData() const;

    const std::vector<double> &getMDepthData() const;

private:
    std::vector<Vector3d> m_rgb_data;
    std::vector<double> m_depth_data;
    std::vector<Vector3d> m_landmarks;
};

#endif //FACE_RECONSTRUCTION_SINGLEINPUTFRAME_H
