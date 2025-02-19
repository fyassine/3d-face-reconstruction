#ifndef FACE_RECONSTRUCTION_SINGLEINPUTFRAME_H
#define FACE_RECONSTRUCTION_SINGLEINPUTFRAME_H

#include "Eigen.h"
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

class SingleInputFrame {
public:
    SingleInputFrame(std::vector<Vector3d> rgbData, std::vector<double> depthData, std::vector<Vector3d> landmarks);
    SingleInputFrame() = default;
    ~SingleInputFrame();
    [[nodiscard]] const std::vector<Vector3d> &getMLandmarks() const;
    [[nodiscard]] const std::vector<Vector3d> &getMRgbData() const;
    [[nodiscard]] const std::vector<double> &getMDepthData() const;

private:
    std::vector<Vector3d> m_rgb_data;
    std::vector<double> m_depth_data;
    std::vector<Vector3d> m_landmarks;

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & archive) {
        archive(m_rgb_data, m_depth_data, m_landmarks);
    }
};

#endif //FACE_RECONSTRUCTION_SINGLEINPUTFRAME_H