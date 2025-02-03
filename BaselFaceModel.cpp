#include "BaselFaceModel.h"

BaselFaceModel::BaselFaceModel(){
    setupFaces();
    setupLandmarkIndices();
    setupHDF5Data();
    initializeParameters();
    computeTransformationMatrix();
    //TODO read hdf5
}

std::vector<Vector3d> BaselFaceModel::getLandmarks() {
    std::vector<Vector3d> landmarks;
    auto vertices = getVerticesWithoutTransformation();
    landmarks.reserve(landmark_indices.size());
    for (int landmark_index : landmark_indices) {
        landmarks.emplace_back(vertices[landmark_index]);
    }
    return landmarks;
}

Vector3d BaselFaceModel::getVertex(int vertexId) {
    std::vector<Vector3d> landmarks;
    auto vertices = getVerticesWithoutTransformation();
    return vertices[vertexId];
}

BaselFaceModel::~BaselFaceModel() = default;
