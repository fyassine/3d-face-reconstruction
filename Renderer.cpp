#include "Renderer.h"

Renderer::Renderer() = default;
Renderer::~Renderer() = default;

void Renderer::run(const std::vector<Vector3d>& modelVertices, const std::vector<Vector3i>& modelColors, const std::vector<int>& modelFaces, const Matrix3d& intrinsicMatrix, const Matrix4d& extrinsicMatrix, const std::string& inputPath, const std::string& outputPath) {
    cv::Mat image = cv::imread(inputPath);

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
        colors.push_back(cv::Scalar(color(2), color(1), color(0)));  // BGR format in OpenCV, vllt. invertieren?
    }
    std::cout << faces.size() << std::endl;

    renderModel(image, vertices, faces, intrinsics, R, t, colors);
    
    // Apply smoothing to the reconstructed face
    //cv::Mat smoothedImage;
    //cv::bilateralFilter(image, smoothedImage, 9, 75, 75);  // (d=9, sigmaColor=75, sigmaSpace=75)

    // Save the filtered image
    cv::imwrite(outputPath, image);
    cv::waitKey(0);
}

void Renderer::renderModel(cv::Mat &image, const std::vector<cv::Point3f> &vertices, const std::vector<cv::Vec3i> &faces,
                           const cv::Mat &intrinsicMatrix, const cv::Mat &R, const cv::Mat &t, const std::vector<cv::Scalar> &colors) {
    std::vector<cv::Point2f> projectedPoints;

    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    cv::projectPoints(vertices, rvec, t, intrinsicMatrix, cv::Mat(), projectedPoints);

    std::vector<std::pair<double, cv::Vec3i>> facesWithDepth;
    for (size_t i = 0; i < faces.size(); ++i) {
        double avgDepth = 0.0;
        for (int j = 0; j < 3; ++j) {
            const auto& vertex = vertices[faces[i][j]];
            avgDepth += vertex.z;
        }
        avgDepth /= 3.0;
        facesWithDepth.push_back({avgDepth, faces[i]});
    }

    std::sort(facesWithDepth.begin(), facesWithDepth.end(),
              [](const std::pair<double, cv::Vec3i>& a, const std::pair<double, cv::Vec3i>& b) {
                  return a.first > b.first;
              });

    for (const auto& faceWithDepth : facesWithDepth) {
        const auto& face = faceWithDepth.second;
        cv::Point pts[3];
        for (int j = 0; j < 3; ++j) {
            pts[j] = projectedPoints[face[j]];
        }
        cv::Scalar faceColor = (colors[face[0]] + colors[face[1]] + colors[face[2]]) / 3;
        cv::fillConvexPoly(image, pts, 3, faceColor);
    }
}

void Renderer::convertPngsToMp4(const std::string &inputPath, const std::string &outputPath, int numberOfFrames) {
    std::string outputDir = "../../../Result/video/";

    // Video Writer setup
    int frameWidth = 1280;
    int frameHeight = 720;
    cv::VideoWriter videoWriter(outputDir + "output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 5, cv::Size(frameWidth, frameHeight));

    if (!videoWriter.isOpened()) {
        std::cerr << "Could not open the output video file for writing" << std::endl;
        return;
    }

    for (int frameIdx = 1; frameIdx < numberOfFrames; ++frameIdx) {  // Example: 100 frames
        cv::Mat frame = cv::imread(inputPath + std::to_string(frameIdx) + ".png");
        videoWriter.write(frame);
    }
    videoWriter.release();
    cv::destroyAllWindows();
}

void Renderer::convertColorToPng(std::vector<Vector3d> colorValues, const std::string& path) {
    const int width = 1280;
    const int height = 720;
    cv::Mat image(height, width, CV_8UC3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            const Vector3d& color = colorValues[index];

            // Convert color values to 8-bit unsigned char
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uchar>(std::clamp(color(2) * 255.0, 0.0, 255.0)),
                    static_cast<uchar>(std::clamp(color(1) * 255.0, 0.0, 255.0)),
                    static_cast<uchar>(std::clamp(color(0) * 255.0, 0.0, 255.0))
            );
        }
    }
    std::cout << "Saved to" << path << std::endl;
    cv::imwrite(path, image);
}

void Renderer::generatePhotometricError(const std::string &inputPath1, const std::string &inputPath2,
                                        const std::string &outputPath) {
    cv::Mat image1 = cv::imread(inputPath1);
    cv::Mat image2 = cv::imread(inputPath2);
    cv::Mat errorMap(image1.size(), CV_8UC3);

    const float maxError = 3.0f;
    float photometricErrorAvg = 0;

    for (int y = 0; y < image1.rows; ++y) {
        for (int x = 0; x < image1.cols; ++x) {
            cv::Vec3b pixel1 = image1.at<cv::Vec3b>(y, x);
            cv::Vec3b pixel2 = image2.at<cv::Vec3b>(y, x);
            float error = cv::norm(pixel1, pixel2, cv::NORM_L2);
            photometricErrorAvg += maxError;
            float t = std::clamp(error / maxError, 0.0f, 1.0f);
            int r = static_cast<int>(255 * t);
            int g = 0;
            int b = static_cast<int>(255 * (1 - t));
            errorMap.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }
    cv::imwrite(outputPath, errorMap);
    std::cout << "Photometric Error: " << photometricErrorAvg/(image1.rows * image1.cols) << std::endl;
}

