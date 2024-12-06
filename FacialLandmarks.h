#ifndef FACE_RECONSTRUCTION_FACIALLANDMARKS_H
#define FACE_RECONSTRUCTION_FACIALLANDMARKS_H

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <iostream>
#include <dlib/image_loader/load_image.h>
#include <dlib/image_saver/save_png.h>

/**
 * @brief Detects facial landmarks in an image.
 *
 * This function detects faces in an image, and for the first detected face,
 * it uses a shape predictor model to return the landmarks.
 *
 * @param imagePath The path to the image file to process. This image should contain a face.
 * @param shapePredictorPath The path to the shape predictor file.
 *
 * @note There is no error handling yet.
 *
 * @return dlib::full_object_detection The detected landmarks for the first face found in the image.
 */
//TODO: Include depth?!
static dlib::full_object_detection GetLandmarks(const std::string& imagePath, const std::string& shapePredictorPath) {
    auto frontal_face_detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor shape_predictor;
    dlib::deserialize(shapePredictorPath) >> shape_predictor;
    dlib::array2d<dlib::rgb_pixel> image;
    dlib::load_image(image, imagePath);
    auto detectedFaces = frontal_face_detector(image);
    return shape_predictor(image, detectedFaces[0]);
}

/**
 * @brief Determines the color for a specific landmark based on its index.
 *
 * This function maps each landmark to a color for visualization purposes.
 * Different facial regions (jaw, eyebrows, nose, eyes, mouth) are assigned different colors.
 *
 * @param i The index of the landmark to get the color for.
 *
 * @return dlib::rgb_pixel The color of the given landmark.
 */
static dlib::rgb_pixel GetLandmarkColor(int i) {
    std::vector<std::pair<std::pair<int, int>, dlib::rgb_pixel>> ranges = {
            {{0, 16}, dlib::rgb_pixel(255, 0, 0)},       // Jaw
            {{17, 26}, dlib::rgb_pixel(255, 255, 0)},    // Eyebrows
            {{27, 35}, dlib::rgb_pixel(0, 0, 255)},      // Nose
            {{36, 47}, dlib::rgb_pixel(0, 255, 255)},    // Eyes
            {{48, 68}, dlib::rgb_pixel(0, 255, 0)}       // Mouth
    };
    for (const auto& range : ranges) {
        if (i >= range.first.first && i <= range.first.second) {
            return range.second;
        }
    }
    return dlib::rgb_pixel(0, 0, 0);
}

/**
 * @brief Draws facial landmarks on the image and saves the result.
 *
 * This function detects the facial landmarks in the input image and draws circles
 * at each landmark position, colored based on the facial region. The resulting
 * image is saved to the specified output file path.
 *
 * @param imagePath The path to the image file to process. This image should contain a face.
 * @param outputPath The path where the image will be saved.
 * @param shapePredictorPath TThe path to the shape predictor file.
 */
static void DrawLandmarksOnImage(const std::string& imagePath, const std::string& outputPath, const std::string& shapePredictorPath){
    auto shape = GetLandmarks(imagePath, shapePredictorPath);
    dlib::array2d<dlib::rgb_pixel> image;
    dlib::load_image(image, imagePath);
    for (int i = 0; i < shape.num_parts(); ++i) {
        int x = shape.part(i).x();
        int y = shape.part(i).y();
        draw_solid_circle(image, dlib::point(x, y), 3, GetLandmarkColor(i));
    }
    dlib::save_png(image, outputPath);
}



#endif //FACE_RECONSTRUCTION_FACIALLANDMARKS_H
