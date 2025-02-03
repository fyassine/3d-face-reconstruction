#include <iostream>
#include "InputDataExtractor.h"

InputDataExtractor::InputDataExtractor() {

}

InputDataExtractor::~InputDataExtractor() = default;

InputData InputDataExtractor::extractInputData(const std::string& path) {
    auto align_to = RS2_STREAM_COLOR;
    rs2::align align(align_to);
    std::vector<SingleInputFrame> frames;

    try {
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_device_from_file(path);
        cfg.enable_stream(RS2_STREAM_COLOR);
        cfg.enable_stream(RS2_STREAM_DEPTH);
        pipe.start(cfg);
        Matrix3d extracted_intrinsic_matrix;
        Matrix4d extracted_extrinsic_matrix;
        int width;
        int height;

        {
            rs2::frameset unaligned_frames;
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

        while (true) {
            rs2::frameset unaligned_frames;
            if (!pipe.poll_for_frames(&unaligned_frames)) {
                break;
            }
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

            std::vector<Vector3d> landmarks; //TODO: Fill landmarks
            frames.emplace_back(rgbData, depthData, landmarks);
        }
        return InputData(frames, width, height, extracted_intrinsic_matrix, extracted_extrinsic_matrix, frames[0]);
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
