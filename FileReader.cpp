#include <iostream>
#include <fstream>
#include "FileReader.h"

FileReader::FileReader() = default;
FileReader::~FileReader() = default;

std::vector<Vector3d> FileReader::read3DVerticesFromTxt(const std::string& path) {
    const std::string inputFile = std::string(dataFolderPath + path);
    std::ifstream inFile(inputFile);
    std::string line;
    std::vector<Eigen::Vector3d> vertices;
    while (std::getline(inFile, line)) {
        std::istringstream iss(line);
        float first, second, third;
        if (iss >> first >> second >> third) {
            Eigen::Vector3f newVertex;
            newVertex.x() = first;
            newVertex.y() = second;
            newVertex.z() = third;
            vertices.emplace_back(newVertex);
        } else {
            std::cerr << "Error: Incorrect input file" << std::endl;
        }
    }
    return vertices;
}
