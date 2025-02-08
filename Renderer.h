#include "Eigen.h"
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

class Renderer {
public:
    Renderer();
    ~Renderer();
    static void renderModel(cv::Mat& image,
                     const std::vector<cv::Point3f>& vertices,
                     const std::vector<cv::Vec3i>& faces,
                     const cv::Mat& intrinsicMatrix,
                     const cv::Mat& R,
                     const cv::Mat& t,
                     const std::vector<cv::Scalar>& colors);
    static void convertPngsToMp4(const std::string& inputPath, const std::string& outputPath, int numberOfFrames);
    static void convertColorToPng(std::vector<Vector3d> colorValues, const std::string& path);
    static void run(const std::vector<Vector3d> &modelVertices, const std::vector<Vector3i> &modelColors,
             const std::vector<int> &modelFaces, const Matrix3d &intrinsicMatrix, const Matrix4d &extrinsicMatrix, const std::string& inputPath, const std::string& outputPath);
};

