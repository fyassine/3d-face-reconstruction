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

InputData::InputData() = default;

InputData::~InputData() = default;

SingleInputFrame* InputData::processNextFrame() {
    if (m_current_frame_index >= m_frames.size()) {
        return nullptr;
    }
    m_current_frame_index++;
    m_currentFrame = m_frames[m_current_frame_index];
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

std::vector<Vector3d> InputData::getAllCorrespondences(std::vector<Vector3d> vertices) {
    auto depthData = m_currentFrame.getMDepthData();
    std::vector<Vector3d> correspondences;
    for (int i = 0; i < vertices.size(); ++i) {
        auto target2D = InputDataExtractor::convert3Dto2D(vertices[i], m_intrinsic_matrix, m_extrinsic_matrix);
        double depthInputImage = depthData[(int) target2D.x() + (int) target2D.y() * m_width];
        auto target = InputDataExtractor::convert2Dto3D(target2D, depthInputImage, m_intrinsic_matrix, m_extrinsic_matrix);
        correspondences.emplace_back(target);
    }
    return correspondences;
}

Vector3d InputData::getCorrespondingPoint(const Vector3d& bfmVertex) {
    auto depthData = m_currentFrame.getMDepthData();
    auto target2D = InputDataExtractor::convert3Dto2D(bfmVertex, m_intrinsic_matrix, m_extrinsic_matrix);
    double depthInputImage = depthData[(int) target2D.x() + (int) target2D.y() * m_width];
    auto target = InputDataExtractor::convert2Dto3D(target2D, depthInputImage, m_intrinsic_matrix, m_extrinsic_matrix);
    return target;
}

std::vector<Vector3i> InputData::getCorrespondingColors(std::vector<Vector3d> vertices) {
    std::vector<Vector3i> colorValues;
    std::vector<Vector3d> colorImage = m_currentFrame.getMRgbData();

    for (int i = 0; i < vertices.size(); ++i) {
        auto coordinate2D = InputDataExtractor::convert3Dto2D(vertices[i], m_intrinsic_matrix, m_extrinsic_matrix);
        Vector3d newColor = colorImage[(int) coordinate2D.y() * (int) m_width + (int) coordinate2D.x()];
        Vector3i newColorInt = Vector3i((int) (newColor.x() * 255), (int) (newColor.y() * 255), (int) (newColor.z() * 255));
        colorValues.emplace_back(newColorInt);
    }
    return colorValues;
}

const std::vector<SingleInputFrame> &InputData::getMFrames() const {
    return m_frames;
}

void InputData::save(const std::string& filename) {
    std::ofstream os(filename);
    cereal::JSONOutputArchive archive(os);
    archive(*this);
}

InputData InputData::load(const std::string& filename) {
    std::ifstream is(filename);
    cereal::JSONInputArchive archive(is);
    InputData data;
    archive(data);
    return data;
}

std::vector<SingleInputFrame> InputData::m_frames1() const
{
    return m_frames;
}
