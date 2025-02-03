#ifndef FACE_RECONSTRUCTION_INPUTDATA_H
#define FACE_RECONSTRUCTION_INPUTDATA_H

#include "SingleInputFrame.h"
#include <vector>

class InputData {
public:
    InputData(std::vector<SingleInputFrame> frames, int width, int height, Matrix3d intrinsic_matrix,
              Matrix4d extrinsic_matrix, const SingleInputFrame& currentFrame);

    ~InputData();
    SingleInputFrame* processNextFrame();
private:
    int m_current_frame_index = 0;
    SingleInputFrame m_currentFrame;
public:
    const SingleInputFrame &getMCurrentFrame() const;

private:
    std::vector<SingleInputFrame> m_frames;

    int m_width;
    int m_height;
    Matrix3d m_intrinsic_matrix;
    Matrix4d m_extrinsic_matrix;
};

#endif //FACE_RECONSTRUCTION_INPUTDATA_H
