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
    outFile << "element face " << faces.size() << std::endl;
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
