#include "SingleInputFrame.h"

#include <utility>

SingleInputFrame::SingleInputFrame(std::vector<Vector3d> rgbData, std::vector<double> depthData,
                                   std::vector<Vector3d> landmarks) {
    m_rgb_data = std::move(rgbData);
    m_depth_data = std::move(depthData);
    m_landmarks = std::move(landmarks);
}

SingleInputFrame::~SingleInputFrame() = default;
