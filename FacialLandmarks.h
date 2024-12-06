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
 * @brief Detects facial landmarks in an image and saves the results. (This is wip and subject to change)
 *
 * This function takes an image and a shape predictor, detects faces in the image, and then identifies
 * facial landmarks such as eyes, nose, mouth, jaw, and eyebrows. The detected landmarks are optionally drawn
 * on the image, and the modified image can be saved if specified.
 *
 * @param imagePath The path to the image file to process. This image should contain a face or faces.
 * @param shapePredictorPath The path to the shape predictor file used to predict facial landmarks.
 * @param saveResult Optional flag to indicate whether to save the modified image with drawn landmarks. Default is false.
 * @param resultPath Optional path where the resulting image with landmarks will be saved. Default is an empty string.
 *
 * @note The landmarks are categorized into different facial regions:
 *       - Jaw (red)
 *       - Eyebrows (yellow)
 *       - Nose (blue)
 *       - Eyes (cyan)
 *       - Mouth (green)
 * @note There is no error handling yet.
 */

static void GetLandmarks(const std::string& imagePath, const std::string& shapePredictorPath, bool saveResult=false, const std::string& resultPath="") {
    auto frontal_face_detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor shape_predictor;
    std::cout << shapePredictorPath << std::endl;

    dlib::deserialize(shapePredictorPath) >> shape_predictor;
    dlib::array2d<dlib::rgb_pixel> image;
    dlib::load_image(image, imagePath);
    auto detectedFaces = frontal_face_detector(image);

    for (int i = 0; i < detectedFaces.size(); ++i) {
        auto shape = shape_predictor(image, detectedFaces[i]);
        for (int j = 0; j < shape.num_parts(); ++j) {
            int x = shape.part(j).x();
            int y = shape.part(j).y();
            if(saveResult){
                auto color = dlib::rgb_pixel(0, 0, 0);
                if(j < 18){
                    color = dlib::rgb_pixel(255, 0, 0); //jaw
                }else if(j < 28){
                    color = dlib::rgb_pixel(255, 255, 0); //eyebrows
                }else if(j < 36){
                    color = dlib::rgb_pixel(0, 0, 255); //nose
                }else if(j < 49){
                    color = dlib::rgb_pixel(0, 255, 255); //eyes
                }else{
                    color = dlib::rgb_pixel(0, 255, 0); //mouth
                }
                draw_solid_circle(image, dlib::point(x, y), 3, color);
            }
        }
    }
    if(saveResult){
        dlib::save_png(image, resultPath);
    }
}

#endif //FACE_RECONSTRUCTION_FACIALLANDMARKS_H
