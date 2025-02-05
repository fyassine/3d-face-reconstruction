#ifndef FACE_RECONSTRUCTION_INPUTDATA_H
#define FACE_RECONSTRUCTION_INPUTDATA_H

#include "SingleInputFrame.h"
#include <vector>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <Eigen/Dense>

namespace cereal {
    // Serialization for Eigen::Matrix3d
    template <class Archive>
    void serialize(Archive& archive, Eigen::Matrix3d& matrix) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                archive(matrix(i, j));
            }
        }
    }

    // Serialization for Eigen::Matrix4d
    template <class Archive>
    void serialize(Archive& archive, Eigen::Matrix4d& matrix) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                archive(matrix(i, j));
            }
        }
    }

    // Serialization for Eigen::Vector3d
    template <class Archive>
    void serialize(Archive& archive, Eigen::Vector3d& vector) {
        archive(vector.x(), vector.y(), vector.z());
    }
}

class InputData {
public:
    InputData(std::vector<SingleInputFrame> frames, int width, int height, Eigen::Matrix3d intrinsic_matrix,
              Eigen::Matrix4d extrinsic_matrix, const SingleInputFrame& currentFrame);

    ~InputData();
    InputData();
    SingleInputFrame* processNextFrame();
    const SingleInputFrame &getMCurrentFrame() const;
    const Eigen::Matrix3d &getMIntrinsicMatrix() const;
    const Eigen::Matrix4d &getMExtrinsicMatrix() const;

    Eigen::Vector3d getCorrespondingPoint(const Eigen::Vector3d& bfmVertex);
    std::vector<Eigen::Vector3i> getCorrespondingColors(std::vector<Eigen::Vector3d> vertices);
    std::vector<Eigen::Vector3d> getAllCorrespondences(std::vector<Eigen::Vector3d> vertices);

    void save(const std::string& filename);
    static InputData load(const std::string& filename);

private:
    int m_current_frame_index = 0;
    SingleInputFrame m_currentFrame;
    std::vector<SingleInputFrame> m_frames;
    int m_width;
    int m_height;
    Eigen::Matrix3d m_intrinsic_matrix;
    Eigen::Matrix4d m_extrinsic_matrix;

    // Serialization function
    template<class Archive>
    void serialize(Archive & archive) {
        archive(m_current_frame_index, m_currentFrame, m_frames, m_width, m_height, m_intrinsic_matrix, m_extrinsic_matrix);
    }

    // cereal::access as a friend to allow serialization of private members
    friend class cereal::access;
};

#endif //FACE_RECONSTRUCTION_INPUTDATA_H