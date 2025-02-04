#include "InputData.h"

#include <utility>
#include "InputDataExtractor.h"

InputData::InputData(std::vector<SingleInputFrame> frames, int width, int height, Matrix3d intrinsic_matrix,
                     Matrix4d extrinsic_matrix, const SingleInputFrame& currentFrame) : m_currentFrame(currentFrame) {
    m_frames = std::move(frames);
    m_width = width;
    m_height = height;
    m_intrinsic_matrix = std::move(intrinsic_matrix);
    m_extrinsic_matrix = std::move(extrinsic_matrix);
}

InputData::~InputData() = default;

SingleInputFrame* InputData::processNextFrame() {
    if(m_current_frame_index >= m_frames.size()){
        return nullptr;
    }
    m_currentFrame = m_frames[m_current_frame_index];
    m_current_frame_index++;
    return &m_currentFrame;
}

const SingleInputFrame &InputData::getMCurrentFrame() const {
    return m_currentFrame;
}

const Matrix3d &InputData::getMIntrinsicMatrix() const {
    return m_intrinsic_matrix;
}

const Matrix4d &InputData::getMExtrinsicMatrix() const {
    return m_extrinsic_matrix;
}

Vector3d InputData::getCorrespondingPoint(const Vector3d& bfmVertex) {
    auto depthData = m_currentFrame.getMDepthData();
    auto target2D = InputDataExtractor::convert3Dto2D(bfmVertex, m_intrinsic_matrix, m_extrinsic_matrix);
    double depthInputImage = depthData[(int) target2D.x() + (int) target2D.y() * m_width];
    auto target = InputDataExtractor::convert2Dto3D(target2D, depthInputImage, m_intrinsic_matrix, m_extrinsic_matrix);
    return target;
}
