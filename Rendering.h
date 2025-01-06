#ifndef FACE_RECONSTRUCTION_RENDERING_H
#define FACE_RECONSTRUCTION_RENDERING_H

#include <dlib/opencv.h>
//#include "opencv2/imgcodecs.hpp"
#include "BFMParameters.h"

/*void renderImage(cv::Mat inputImage){
    cv::Mat renderedImage;
    cv::imwrite("result.png", renderedImage);
}*/

static BfmProperties getProperties(const std::string& path){
    BfmProperties properties;
    initializeBFM(path, properties);
    return properties;
}

static void convertParametersToPly(const BfmProperties& properties, const std::string& resultPath){

    std::ofstream outFile(resultPath);
    //Header
    outFile << "ply" << std::endl;
    outFile << "format ascii 1.0" << std::endl;
    outFile << "element vertex " << properties.numberOfVertices << std::endl;
    outFile << "property float x" << std::endl;
    outFile << "property float y" << std::endl;
    outFile << "property float z" << std::endl;
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
    outFile << "property uchar alpha" << std::endl;
    outFile << "element face " << properties.numberOfTriangles << std::endl;
    outFile << "property list uchar int vertex_indices" << std::endl;
    outFile << "end_header" << std::endl;
    //Vertices
    auto vertices = getVertices(properties);
    auto colorValues = getColorValues(properties);
    for (int i = 0; i < properties.numberOfVertices; i++) {
        //Position
        auto x = vertices[i].x();
        auto y = vertices[i].y();
        auto z = vertices[i].z();
        //Color
        auto r = colorValues[i].x();
        auto g = colorValues[i].y();
        auto b = colorValues[i].z();
        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }
    //Faces
    for (int i = 0; i < properties.numberOfTriangles * 3; i+=3) {
        outFile << "3 " << properties.triangles[i] << " " << properties.triangles[i + 1] << " " << properties.triangles[i + 2] << std::endl;
    }
    outFile.close();
}

#endif //FACE_RECONSTRUCTION_RENDERING_H
