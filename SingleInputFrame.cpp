#include "SingleInputFrame.h"

#include <utility>

SingleInputFrame::SingleInputFrame(std::vector<Vector3d> rgbData, std::vector<double> depthData,
                                   std::vector<Vector3d> landmarks) {
    m_rgb_data = std::move(rgbData);
    m_depth_data = std::move(depthData);
    m_landmarks = std::move(landmarks);
}

const std::vector<Vector3d> &SingleInputFrame::getMLandmarks() const {
    return m_landmarks;
}

const std::vector<Vector3d> &SingleInputFrame::getMRgbData() const {
    return m_rgb_data;
}

const std::vector<double> &SingleInputFrame::getMDepthData() const {
    return m_depth_data;
}

SingleInputFrame::~SingleInputFrame() = default;
