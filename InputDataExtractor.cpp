#include <iostream>
#include <FreeImage.h>
#include "InputDataExtractor.h"
#include "FacialLandmarks.h"

#define NUMBER_OF_FRAMES 1

InputDataExtractor::InputDataExtractor() = default;

InputDataExtractor::~InputDataExtractor() = default;

InputData InputDataExtractor::extractInputData(const std::string& path) {
    auto align_to = RS2_STREAM_COLOR;
    rs2::align align(align_to);
    std::vector<SingleInputFrame> frames;

    try {
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_device_from_file(DATA_FOLDER_PATH + path);
        cfg.enable_stream(RS2_STREAM_COLOR);
        cfg.enable_stream(RS2_STREAM_DEPTH);
        pipe.start(cfg);
        Matrix3d extracted_intrinsic_matrix;
        Matrix4d extracted_extrinsic_matrix;
        int width;
        int height;

        {
            rs2::frameset unaligned_frames = pipe.wait_for_frames();  // Ensure we get frames
            rs2::frameset frameset = align.process(unaligned_frames);

            auto depth = frameset.get_depth_frame();
            auto color = frameset.get_color_frame();

            auto depth_profile = depth.get_profile().as<rs2::video_stream_profile>();
            rs2_intrinsics intrinsics = depth_profile.get_intrinsics();
            auto extrinsics = depth_profile.get_extrinsics_to(color.get_profile().as<rs2::video_stream_profile>());

            width = intrinsics.width;
            height = intrinsics.height;

            extracted_intrinsic_matrix << intrinsics.fx, 0, intrinsics.ppx,
                    0, intrinsics.fy, intrinsics.ppy,
                    0, 0, 1;

            extracted_extrinsic_matrix = Matrix4d::Identity();
            for (int i = 0; i < 9; i++) {
                extracted_extrinsic_matrix(i / 3, i % 3) = extrinsics.rotation[i];
            }
            for (int i = 0; i < 3; i++) {
                extracted_extrinsic_matrix(i, 3) = extrinsics.translation[i];
            }
        }


        {
            int counter = 0;
            while (NUMBER_OF_FRAMES > counter) {
                counter++;
                rs2::frameset unaligned_frames = pipe.wait_for_frames();
                // if (!pipe.poll_for_frames(&unaligned_frames)) {
                //     break;
                // }
                rs2::frameset frameset = align.process(unaligned_frames);

                auto depth = frameset.get_depth_frame();
                auto color = frameset.get_color_frame();

                std::vector<double> depthData(width * height);
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        depthData[i * width + j] = depth.get_distance(j, i);
                    }
                }

                std::vector<Vector3d> rgbData(width * height);
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        uint8_t* color_pixel = (uint8_t*)color.get_data() + (i * color.get_stride_in_bytes()) + j * 3;
                        rgbData[i * width + j] = Vector3d(color_pixel[0] / 255.0, color_pixel[1] / 255.0, color_pixel[2] / 255.0);
                    }
                }

                convertVideoFrameToPng(color, "../../../Result/color_frame_for_landmark_detection.png");
                std::vector<Vector3d> landmarks = searchForLandmarks(depthData, extracted_intrinsic_matrix, extracted_extrinsic_matrix);
                frames.emplace_back(rgbData, depthData, landmarks);
            }
        }
        return {frames, width, height, extracted_intrinsic_matrix, extracted_extrinsic_matrix, frames[0]};
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return {frames, 0, 0, Matrix3d::Zero(), Matrix4d::Zero(), frames[0]};
}

void InputDataExtractor::convertVideoFrameToPng(rs2::video_frame videoFrame, std::string name) {
    int width = videoFrame.get_width();
    int height = videoFrame.get_height();
    const uint8_t* data = static_cast<const uint8_t*>(videoFrame.get_data());

    std::vector<uint8_t> bgr_data(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        bgr_data[i * 3 + 0] = data[i * 3 + 2]; // B
        bgr_data[i * 3 + 1] = data[i * 3 + 1]; // G
        bgr_data[i * 3 + 2] = data[i * 3 + 0]; // R
    }

    FreeImage_Initialise();

    FIBITMAP* bitmap = FreeImage_ConvertFromRawBits(
            bgr_data.data(), width, height, width * 3, 24,
            0x0000FF, 0x00FF00, 0xFF0000, true
    );

    if (bitmap) {
        std::string outputFileName = "../../../Result/color_frame_for_landmark_detection.png";
        if (FreeImage_Save(FIF_PNG, bitmap, outputFileName.c_str())) {
            std::cout << "Saved color frame to " << outputFileName << std::endl;
        } else {
            std::cerr << "Failed to save the color frame." << std::endl;
        }
        FreeImage_Unload(bitmap);
    } else {
        std::cerr << "Failed to create bitmap from color frame." << std::endl;
    }
    FreeImage_DeInitialise();
}

std::vector<Vector3d> InputDataExtractor::searchForLandmarks(std::vector<double> depthValues, const Matrix3d& intrinsics, const Matrix4d& extrinsics) {
    auto landmarks2D = GetLandmarkVector("../../../Result/color_frame_for_landmark_detection.png", "../../../Data/shape_predictor_68_face_landmarks.dat");
    std::vector<Vector3d> landmarks;
    for (int i = 0; i < 68; ++i) {
        Eigen::Vector2d landmark = landmarks2D[i];
        int pixel_x = (int) landmark.x();
        int pixel_y = (int) landmark.y();
        double depth_value = depthValues[pixel_y * 1280 + pixel_x];
        //TODO: Correct Depth Value OR set landmark to -1, -1, -1
        landmarks.emplace_back(convert2Dto3D(landmark, depth_value, intrinsics, extrinsics));
    }
    return landmarks;
}

Vector3d InputDataExtractor::convert2Dto3D(const Vector2d &point, double depth, const Matrix3d &intrinsics,
                                           const Matrix4d &extrinsics) {
    double fX = intrinsics(0, 0);
    double fY = intrinsics(1, 1);
    double cX = intrinsics(0, 2);
    double cY = intrinsics(1, 2);

    double x = (point.x() - cX) * depth / fX;
    double y = (point.y() - cY) * depth / fY;
    double z = depth;

    Matrix4d depthExtrinsicsInv = extrinsics.inverse();

    Vector4d cameraCoord = Vector4d(x, y, z, 1.0f);
    Vector4d worldCoords = depthExtrinsicsInv * cameraCoord;

    return {worldCoords.x(), worldCoords.y(), worldCoords.z()};
}

Vector2d InputDataExtractor::convert3Dto2D(const Eigen::Vector3d& point, const Eigen::Matrix3d& depthIntrinsics, const Eigen::Matrix4d& extrinsics) {
    Eigen::Matrix4d depthExtrinsicsInv = extrinsics.inverse();
    Eigen::Vector4d worldCoord(point.x(), point.y(), point.z(), 1.0);
    Eigen::Vector4d cameraCoord = depthExtrinsicsInv * worldCoord;

    double fX = depthIntrinsics(0, 0); // focal length in x direction
    double fY = depthIntrinsics(1, 1); // focal length in y direction
    double cX = depthIntrinsics(0, 2); // optical center in x direction
    double cY = depthIntrinsics(1, 2); // optical center in y direction

    double x = cameraCoord.x() / cameraCoord.z();
    double y = cameraCoord.y() / cameraCoord.z();

    double u = fX * x + cX;
    double v = fY * y + cY;

    return Vector2d{u, v};
}

