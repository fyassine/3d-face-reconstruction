#ifndef FACE_RECONSTRUCTION_INPUTDATA_H
#define FACE_RECONSTRUCTION_INPUTDATA_H

#include "SingleInputFrame.h"
#include <vector>

class InputData {
public:
    InputData();
    InputData(std::vector<SingleInputFrame> frames, int width, int height, Matrix3d intrinsic_matrix,
              Matrix4d extrinsic_matrix, const SingleInputFrame& currentFrame);

    ~InputData();
    SingleInputFrame* processNextFrame();
    const SingleInputFrame &getMCurrentFrame() const;
    const Matrix3d &getMIntrinsicMatrix() const;
    const Matrix4d &getMExtrinsicMatrix() const;

    Vector3d getCorrespondingPoint(const Vector3d& bfmVertex);
    std::vector<Vector3i> getCorrespondingColors(std::vector<Vector3d> vertices);
    std::vector<Vector3d> getAllCorrespondences(std::vector<Vector3d> vertices);

private:
    int m_current_frame_index = 0;
    SingleInputFrame m_currentFrame;
    std::vector<SingleInputFrame> m_frames;
    int m_width;
    int m_height;
    Matrix3d m_intrinsic_matrix;
    Matrix4d m_extrinsic_matrix;
};

#endif //FACE_RECONSTRUCTION_INPUTDATA_H
