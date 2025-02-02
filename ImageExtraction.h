#ifndef FACE_RECONSTRUCTION_IMAGEEXTRACTION_H
#define FACE_RECONSTRUCTION_IMAGEEXTRACTION_H
#include "FreeImage.h"
#include "iostream"
#include <librealsense2/rs.hpp>

//TODO: Conversion 2d->3d from bfmparameters to here

struct InputImage{
    int width;
    int height;
    std::vector<Eigen::Vector3f> color;
    std::vector<float> depthValues;
    std::vector<Eigen::Vector2f> landmarks;
    std::vector<float> depthValuesLandmarks;
    Matrix3f intrinsics;
    Matrix4f extrinsics;
};

static void printInputImage(const InputImage& inputImage) {
    // Print width and height
    // Print color
    if (inputImage.color.empty()) {
        std::cout << "Color data: Not initialized" << std::endl;
    } else {
        std::cout << "Color data (size " << inputImage.color.size() << "):" << std::endl;
        //for (const auto& color : inputImage.color) {
        //    std::cout << "  (" << color.x() << ", " << color.y() << ", " << color.z() << ")" << std::endl;
        //}
    }

    // Print depth values
    if (inputImage.depthValues.empty()) {
        std::cout << "Depth values: Not initialized" << std::endl;
    } else {
        std::cout << "Depth values (size " << inputImage.depthValues.size() << "):" << std::endl;
        //for (float depth : inputImage.depthValues) {
        //    std::cout << "  " << depth << std::endl;
        //}
    }

    // Print landmarks
    if (inputImage.landmarks.empty()) {
        std::cout << "Landmarks: Not initialized" << std::endl;
    } else {
        std::cout << "Landmarks (size " << inputImage.landmarks.size() << "):" << std::endl;
        for (const auto& landmark : inputImage.landmarks) {
            std::cout << "  (" << landmark.x() << ", " << landmark.y() << ")" << std::endl;
        }
    }

    // Print depth values for landmarks
    if (inputImage.depthValuesLandmarks.empty()) {
        std::cout << "Depth values for landmarks: Not initialized" << std::endl;
    } else {
        std::cout << "Depth values for landmarks (size " << inputImage.depthValuesLandmarks.size() << "):" << std::endl;
        for (float depth : inputImage.depthValuesLandmarks) {
            std::cout << "  " << depth << std::endl;
        }
    }

    // Print intrinsics
    if (inputImage.intrinsics.isZero(1e-6)) {
        std::cout << "Intrinsics: Not initialized" << std::endl;
    } else {
        std::cout << "Intrinsics:" << std::endl;
        std::cout << inputImage.intrinsics << std::endl;
    }

    // Print extrinsics
    if (inputImage.extrinsics.isZero(1e-6)) {
        std::cout << "Extrinsics: Not initialized" << std::endl;
    } else {
        std::cout << "Extrinsics:" << std::endl;
        std::cout << inputImage.extrinsics << std::endl;
    }
}

static std::vector<Eigen::Vector2i> getRelevantKernelPixels(Eigen::Vector2f coordinate, int step){
    std::vector<Eigen::Vector2i> pixelCoordinates;
    //Top + top corners
    for (int i = -step; i < step + 1; ++i) {
        Eigen::Vector2i newCoordinate((int) coordinate.x() + i, (int) coordinate.y() - step);
        pixelCoordinates.emplace_back(newCoordinate);
    }
    //Bottom + bottom corners
    for (int i = -step; i < step + 1; ++i) {
        Eigen::Vector2i newCoordinate((int) coordinate.x() + i, (int) coordinate.y() + step);
        pixelCoordinates.emplace_back(newCoordinate);
    }
    //Left
    for (int i = -step + 1; i < step; ++i) {
        Eigen::Vector2i newCoordinate((int) coordinate.x() - step, (int) coordinate.y() + i);
        pixelCoordinates.emplace_back(newCoordinate);
    }
    //Right
    for (int i = -step + 1; i < step; ++i) {
        Eigen::Vector2i newCoordinate((int) coordinate.x() + step, (int) coordinate.y() + i);
        pixelCoordinates.emplace_back(newCoordinate);
    }
    return pixelCoordinates;
}

static float useKernel(const InputImage& inputImage, const Eigen::Vector2f& coordinate, float avg, float stdDev){
    //Ignore values out of bounds
    int distance = 1; //offset for x and y
    float currentDepth = 1000000.0f;
    Eigen::Vector2f currentCoordinate;
    int currentStepSize = 0;

    while(abs(avg - currentDepth) > stdDev){
        currentStepSize++;
        auto coordinates = getRelevantKernelPixels(coordinate, currentStepSize);
        for (int i = 0; i < coordinates.size(); ++i) {
            if(coordinates[i].y() < 0 || coordinates[i].y() > inputImage.height - 1 || coordinates[i].x() < 0 || coordinates[i].x() > inputImage.width - 1){
                continue;
            }
            float depth = inputImage.depthValues[coordinates[i].y() * inputImage.width + coordinates[i].x()];
            if(depth < 0.0001) {
                continue;
            }
            currentDepth = depth < currentDepth ? depth : currentDepth;
            if(abs(avg - currentDepth) < stdDev){
                std::cout << currentDepth << std::endl;
                return currentDepth;
            }
        }
    }
    return currentDepth;
}

static void correctDepthOfLandmarks(InputImage& inputImage){
    float avg = 0;
    unsigned int n = 17;//inputImage.depthValuesLandmarks.size(); //n = 17?! only jaw
    for (int i = 0; i < (int) n; ++i) {
        avg += inputImage.depthValuesLandmarks[i];
    }
    avg /= (float) n;

    float stdDev = 0;
    for (int i = 0; i < (int) n; ++i) {
        stdDev += powf(inputImage.depthValuesLandmarks[i] - avg, 2);
    }
    stdDev /= (float) n;
    stdDev = sqrtf(stdDev);

    std::cout << "Start Correction" << std::endl;
    //stdDev has to be bigger!!! Otherwise there would be flagged depth values in a perfect model
    //Shouldn't I use abs(avg) and abs(inputImage.depthValuesLandmarks[i]) as a negative depth value might otherwise lead to a wrong result
    for (int i = 0; i < (int) n; ++i) {
        if(abs(avg - inputImage.depthValuesLandmarks[i]) > stdDev){
            std::cout << i << std::endl;
            inputImage.depthValuesLandmarks[i] = inputImage.depthValuesLandmarks[i - 1];//useKernel(inputImage, inputImage.landmarks[i], avg, stdDev);
        }
    }
    std::cout << "End Correction" << std::endl;
}

static void calculateDepthValuesLandmarks(InputImage& inputImage){
    std::cout << "Hey: " << inputImage.landmarks.size() << std::endl;
    for (int i = 0; i < inputImage.landmarks.size(); ++i) {
        Eigen::Vector2f landmark = inputImage.landmarks[i];
        int pixel_x = (int) landmark.x();
        int pixel_y = (int) landmark.y();
        float depth_value = inputImage.depthValues[pixel_y * inputImage.width + pixel_x];
        inputImage.depthValuesLandmarks.emplace_back(depth_value);
    }
    correctDepthOfLandmarks(inputImage);
}

static void writeColorToPng(rs2::video_frame color){
    int width = color.get_width();
    int height = color.get_height();
    const uint8_t* data = static_cast<const uint8_t*>(color.get_data());

    // Convert RGB to BGR
    std::vector<uint8_t> bgr_data(width * height * 3); // Allocate storage for BGR data
    for (int i = 0; i < width * height; ++i) {
        bgr_data[i * 3 + 0] = data[i * 3 + 2]; // B
        bgr_data[i * 3 + 1] = data[i * 3 + 1]; // G
        bgr_data[i * 3 + 2] = data[i * 3 + 0]; // R
    }

    // Initialize FreeImage
    FreeImage_Initialise();

    // Create a FreeImage bitmap
    FIBITMAP* bitmap = FreeImage_ConvertFromRawBits(
            bgr_data.data(), width, height, width * 3, 24,
            0x0000FF, 0x00FF00, 0xFF0000, true // Now using BGR data and flipping vertically
    );

    if (bitmap) {
        // Save the bitmap as a PNG file
        std::string outputFileName = "../../../Result/color_frame_corrected.png";
        if (FreeImage_Save(FIF_PNG, bitmap, outputFileName.c_str())) {
            std::cout << "Saved color frame to " << outputFileName << std::endl;
        } else {
            std::cerr << "Failed to save the color frame." << std::endl;
        }

        // Free the bitmap
        FreeImage_Unload(bitmap);
    } else {
        std::cerr << "Failed to create bitmap from color frame." << std::endl;
    }

    // Deinitialize FreeImage
    FreeImage_DeInitialise();

}


static void writeDepthToPng(rs2::depth_frame depth){
    int width = depth.get_width();
    int height = depth.get_height();
    const uint8_t* data = static_cast<const uint8_t*>(depth.get_data());

    // Convert RGB to BGR
    std::vector<uint8_t> bgr_data(width * height * 3); // Allocate storage for BGR data
    for (int i = 0; i < width * height; ++i) {
        int x = i % width;
        int y = i / width;
        auto color = depth.get_distance(x, y) * 5;
        bgr_data[i * 3 + 0] = color; // B
        bgr_data[i * 3 + 1] = color; // G
        bgr_data[i * 3 + 2] = color; // R
    }

    // Initialize FreeImage
    FreeImage_Initialise();

    // Create a FreeImage bitmap
    FIBITMAP* bitmap = FreeImage_ConvertFromRawBits(
            bgr_data.data(), width, height, width * 3, 24,
            0x0000FF, 0x00FF00, 0xFF0000, true // Now using BGR data and flipping vertically
    );

    if (bitmap) {
        // Save the bitmap as a PNG file
        std::string outputFileName = "../../../Result/greyscale.png";
        if (FreeImage_Save(FIF_PNG, bitmap, outputFileName.c_str())) {
            std::cout << "Saved color frame to " << outputFileName << std::endl;
        } else {
            std::cerr << "Failed to save the color frame." << std::endl;
        }

        // Free the bitmap
        FreeImage_Unload(bitmap);
    } else {
        std::cerr << "Failed to create bitmap from color frame." << std::endl;
    }
    // Deinitialize FreeImage
    FreeImage_DeInitialise();
}

static void writeDepthToPngFromFloat(const InputImage& inputImage){
    int width = inputImage.width;
    int height = inputImage.height;

    // Convert RGB to BGR
    std::vector<uint8_t> bgr_data(width * height * 3); // Allocate storage for BGR data
    for (int i = 0; i < width * height; ++i) {
        int x = i % width;
        int y = i / width;
        auto color = inputImage.depthValues[i] * 5;
        bgr_data[i * 3 + 0] = color; // B
        bgr_data[i * 3 + 1] = color; // G
        bgr_data[i * 3 + 2] = color; // R
    }

    // Initialize FreeImage
    FreeImage_Initialise();

    // Create a FreeImage bitmap
    FIBITMAP* bitmap = FreeImage_ConvertFromRawBits(
            bgr_data.data(), width, height, width * 3, 24,
            0x0000FF, 0x00FF00, 0xFF0000, true // Now using BGR data and flipping vertically
    );

    if (bitmap) {
        // Save the bitmap as a PNG file
        std::string outputFileName = "../../../Result/greyscaleFromFloat.png";
        if (FreeImage_Save(FIF_PNG, bitmap, outputFileName.c_str())) {
            std::cout << "Saved color frame to " << outputFileName << std::endl;
        } else {
            std::cerr << "Failed to save the color frame." << std::endl;
        }

        // Free the bitmap
        FreeImage_Unload(bitmap);
    } else {
        std::cerr << "Failed to create bitmap from color frame." << std::endl;
    }
    // Deinitialize FreeImage
    FreeImage_DeInitialise();
}

static InputImage readVideoData(std::string path){

    InputImage inputImage;
    auto align_to = RS2_STREAM_COLOR;
    rs2::align align(align_to);
    try {
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_device_from_file(path);

        // Optionally configure specific streams
        cfg.enable_stream(RS2_STREAM_COLOR);
        cfg.enable_stream(RS2_STREAM_DEPTH);

        pipe.start(cfg);

        // Wait a moment for frames to populate
        //std::this_thread::sleep_for(std::chrono::milliseconds(50));

        rs2::frameset unaligned_frames;
        rs2::frameset frames;
        unaligned_frames = pipe.wait_for_frames();  // Blocking call to get frames
        frames = align.process(unaligned_frames);

        // Get depth and color frames
        auto depth = frames.get_depth_frame();
        auto color = frames.get_color_frame();

        writeColorToPng(color);
        //writeDepthToPng(depth);
        // Retrieve camera intrinsics for the depth frame
        rs2::video_stream_profile depth_profile = depth.get_profile().as<rs2::video_stream_profile>();
        rs2_intrinsics intrinsics = depth_profile.get_intrinsics();
        auto ex = depth_profile.get_extrinsics_to(color.get_profile().as<rs2::video_stream_profile>());

        // Exit after processing one set of frames
        inputImage.width = intrinsics.width;
        inputImage.height = intrinsics.height;

        // Set intrinsics as an Eigen::Matrix3f
        std::cout << "FX: " << intrinsics.fx << std::endl;
        std::cout << "FY: " << intrinsics.fy << std::endl;
        std::cout << "PPX: " << intrinsics.ppx << std::endl;
        std::cout << "PPY: " << intrinsics.ppy << std::endl;
        std::cout << "Width: " << intrinsics.width << std::endl;
        std::cout << "Height: " << intrinsics.height << std::endl;
        inputImage.intrinsics << intrinsics.fx, 0, intrinsics.ppx,
                0, intrinsics.fy, intrinsics.ppy,
                0, 0, 1;

        // Extract depth values from depth frame
        inputImage.depthValues.resize(inputImage.width * inputImage.height);
        for (int i = 0; i < inputImage.height; i++) {
            for (int j = 0; j < inputImage.width; j++) {
                float depth_value = depth.get_distance(j, i); // Get depth at pixel (j, i)
                inputImage.depthValues[i * inputImage.width + j] = depth_value;
            }
        }

        // Extract color data from color frame
        inputImage.color.resize(inputImage.width * inputImage.height);
        for (int i = 0; i < inputImage.height; i++) {
            for (int j = 0; j < inputImage.width; j++) {
                // Get the color at pixel (j, i)
                uint8_t *color_pixel = (uint8_t*)color.get_data() + (i * color.get_stride_in_bytes()) + j * 3;
                float r = color_pixel[0] / 255.0f;  // Normalize color values
                float g = color_pixel[1] / 255.0f;
                float b = color_pixel[2] / 255.0f;
                inputImage.color[i * inputImage.width + j] = Eigen::Vector3f(r, g, b);
            }
        }

        // Print extrinsics between depth and color streams
        auto depth_to_color = depth_profile.get_extrinsics_to(color.get_profile());
        Eigen::Matrix4f extrinsics = Eigen::Matrix4f::Identity();
        for (int i = 0; i < 9; i++) {
            extrinsics(i / 3, i % 3) = depth_to_color.rotation[i]; // Fill rotation matrix
        }
        for (int i = 0; i < 3; i++) {
            extrinsics(i, 3) = depth_to_color.translation[i]; // Fill translation vector
        }
        inputImage.extrinsics = extrinsics;

    } catch (const rs2::error& e) {
        std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    writeDepthToPngFromFloat(inputImage);
    return inputImage;
}

#endif //FACE_RECONSTRUCTION_IMAGEEXTRACTION_H
