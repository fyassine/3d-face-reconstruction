#include "Renderer.h"

Renderer::~Renderer() = default;

void Renderer::run(const std::vector<Vector3d>& modelVertices, const std::vector<Vector3i>& modelColors, const std::vector<int>& modelFaces, const Matrix3d& intrinsicMatrix, const Matrix4d& extrinsicMatrix) {
    cv::Mat image = cv::imread("../../../Result/color_frame_for_landmark_detection.png");

    cv::Mat intrinsics = (cv::Mat_<double>(3,3) << intrinsicMatrix(0,0), intrinsicMatrix(0,1), intrinsicMatrix(0,2),
            intrinsicMatrix(1,0), intrinsicMatrix(1,1), intrinsicMatrix(1,2),
            intrinsicMatrix(2,0), intrinsicMatrix(2,1), intrinsicMatrix(2,2));
    cv::Mat R = (cv::Mat_<double>(3,3) << extrinsicMatrix(0,0), extrinsicMatrix(0,1), extrinsicMatrix(0,2),
            extrinsicMatrix(1,0), extrinsicMatrix(1,1), extrinsicMatrix(1,2),
            extrinsicMatrix(2,0), extrinsicMatrix(2,1), extrinsicMatrix(2,2));
    cv::Mat t = (cv::Mat_<double>(3,1) << extrinsicMatrix(0,3), extrinsicMatrix(1,3), extrinsicMatrix(2,3));

    std::vector<cv::Point3f> vertices;
    for (const auto& v : modelVertices) {
        vertices.push_back(cv::Point3f(v(0), v(1), v(2)));
    }

    std::vector<cv::Vec3i> faces;
    for (size_t i = 0; i < modelFaces.size(); i += 3) {
        faces.push_back(cv::Vec3i(modelFaces[i], modelFaces[i+1], modelFaces[i+2]));
    }

    std::vector<cv::Scalar> colors;
    for (const auto& color : modelColors) {
        colors.push_back(cv::Scalar(color(0), color(1), color(2)));  // BGR format in OpenCV, vllt. invertieren?
    }

    renderModel(image, vertices, faces, intrinsics, R, t, colors);

    cv::imshow("Rendered Image", image);
    cv::waitKey(0);
}

void Renderer::renderModel(cv::Mat &image, const std::vector<cv::Point3f> &vertices, const std::vector<cv::Vec3i> &faces,
                      const cv::Mat &intrinsicMatrix, const cv::Mat &R, const cv::Mat &t, const std::vector<cv::Scalar> &colors) {
    std::vector<cv::Point2f> projectedPoints;

    // Project 3D points to 2D
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);  // Convert rotation matrix to vector
    cv::projectPoints(vertices, rvec, t, intrinsicMatrix, cv::Mat(), projectedPoints);

    // Draw each face
    for (size_t i = 0; i < faces.size(); ++i) {
        cv::Point pts[3];
        for (int j = 0; j < 3; ++j) {
            pts[j] = projectedPoints[faces[i][j]];
        }
        cv::fillConvexPoly(image, pts, 3, colors[i]);
    }
}

Renderer::Renderer() {

}
