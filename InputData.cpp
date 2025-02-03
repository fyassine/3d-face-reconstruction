#include "InputData.h"

#include <utility>

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
    m_current_frame_index++;
    return &m_frames[m_current_frame_index - 1];
}