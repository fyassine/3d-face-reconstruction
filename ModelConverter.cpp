#include <fstream>
#include "ModelConverter.h"

ModelConverter::ModelConverter() = default;

ModelConverter::~ModelConverter() = default;

void ModelConverter::convertToPly(std::vector<Vector3d> vertices, const std::vector<Vector3i>& color, const std::vector<int>& faces, const std::string& path) {
    std::ofstream outFile(RESULT_FOLDER_PATH + path);

    outFile << "ply" << std::endl;
    outFile << "format ascii 1.0" << std::endl;
    outFile << "element vertex " << vertices.size() << std::endl;
    outFile << "property float x" << std::endl;
    outFile << "property float y" << std::endl;
    outFile << "property float z" << std::endl;
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
    outFile << "property uchar alpha" << std::endl;
    outFile << "element face " << faces.size() / 3 << std::endl;
    outFile << "property list uchar int vertex_indices" << std::endl;
    outFile << "end_header" << std::endl;

    for (int i = 0; i < vertices.size(); i++) {
        auto x = vertices[i].x();
        auto y = vertices[i].y();
        auto z = vertices[i].z();
        auto r = color[i].x();
        auto g = color[i].y();
        auto b = color[i].z();
        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }

    for (int i = 0; i < faces.size(); i+=3) {
        outFile << "3 " << faces[i] << " " << faces[i + 1] << " " << faces[i + 2] << std::endl;
    }
    outFile.close();
}

void ModelConverter::convertToPly(std::vector<Vector3d> vertices, const std::string &path) {
    std::ofstream outFile(RESULT_FOLDER_PATH + path);

    outFile << "ply" << std::endl;
    outFile << "format ascii 1.0" << std::endl;
    outFile << "element vertex " << vertices.size() << std::endl;
    outFile << "property float x" << std::endl;
    outFile << "property float y" << std::endl;
    outFile << "property float z" << std::endl;
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
    outFile << "property uchar alpha" << std::endl;
    outFile << "element face " << 0 << std::endl;
    outFile << "property list uchar int vertex_indices" << std::endl;
    outFile << "end_header" << std::endl;

    for (int i = 0; i < vertices.size(); i++) {
        auto x = vertices[i].x();
        auto y = vertices[i].y();
        auto z = vertices[i].z();
        auto r = 0;
        auto g = 255;
        auto b = 0;
        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }
    outFile.close();
}

void ModelConverter::convertImageToPly(const std::vector<double>& depth, const std::vector<Vector3d>& colorValues, const std::string& path,
                                       const Matrix3d& intrinsics, const Matrix4d& extrinsics) {

    std::vector<Eigen::Vector2d> pointCloudVertices;
    for (int y = 0; y < 720; ++y) {
        for (int x = 0; x < 1280; ++x) {
            pointCloudVertices.emplace_back(x, y);
        }
    }

    std::ofstream outFile(RESULT_FOLDER_PATH + path);
    outFile << "ply" << std::endl;
    outFile << "format ascii 1.0" << std::endl;
    outFile << "element vertex " << pointCloudVertices.size() << std::endl;
    outFile << "property float x" << std::endl;
    outFile << "property float y" << std::endl;
    outFile << "property float z" << std::endl;
    outFile << "property uchar red" << std::endl;
    outFile << "property uchar green" << std::endl;
    outFile << "property uchar blue" << std::endl;
    outFile << "property uchar alpha" << std::endl;
    outFile << "element face " << 0 << std::endl;
    outFile << "property list uchar int vertex_indices" << std::endl;
    outFile << "end_header" << std::endl;

    for (int i = 0; i < pointCloudVertices.size(); i++) {
        Eigen::Vector3d vertex3D = InputDataExtractor::convert2Dto3D(pointCloudVertices[i], depth[i], intrinsics, extrinsics);
        auto x = (float) vertex3D.x();
        auto y = (float) vertex3D.y();
        auto z = (float) vertex3D.z();
        auto r = colorValues[i].x() * 255;
        auto g = colorValues[i].y() * 255;
        auto b = colorValues[i].z() * 255;
        outFile << x << " " << y << " " << z << " " << r << " "<< g << " "<< b << " 255"<< std::endl;
    }
    outFile.close();
}

void ModelConverter::generateGeometricErrorModel(BaselFaceModel *bfm, InputData *inputData, const std::string& path) {
    auto vertices = bfm->transformVertices(bfm->getVerticesWithoutTransformation());
    auto correspondences = inputData->getAllCorrespondences(vertices);
    std::vector<Vector3i> colorValues;
    const float maxDistance = 0.005f;
    float distanceSum = 0;
    for (int i = 0; i < vertices.size(); ++i) {
        float distance = (vertices[i] - correspondences[i]).norm();
        distanceSum += distance;
        float t = std::clamp(distance / maxDistance, 0.0f, 1.0f);
        int r, g, b;

        // If t < 0.5, interpolate between red (255, 0, 0) and green (0, 255, 0)
        if (t < 0.5f) {
            b = static_cast<int>(255 * (1 - 2 * t));  // Red decreases from 255 to 0
            g = static_cast<int>(255 * (2 * t));      // Green increases from 0 to 255
            r = 0;                                    // No blue
        }
            // If t >= 0.5, interpolate between green (0, 255, 0) and blue (0, 0, 255)
        else {
            b = 0;                                    // No red
            g = static_cast<int>(255 * (2 - 2 * t));  // Green decreases from 255 to 0
            r = static_cast<int>(255 * (2 * t - 1));  // Blue increases from 0 to 255
        }

        colorValues.emplace_back(r, g, b);
    }
    std::cout << "GeometricDistance (Mean): " << distanceSum / vertices.size() << std::endl;
    convertToPly(vertices, colorValues, bfm->getFaces(), path);
    //TODO: print mean and std deviations
}

void ModelConverter::convertBFMToPly(BaselFaceModel *bfm, const std::string& path) {
    auto color = bfm->getColorValues();
    auto vertices = bfm->transformVertices(bfm->getVerticesWithoutTransformation());
    auto faces = bfm->getFaces();
    convertToPly(vertices, color, faces, path);
}
