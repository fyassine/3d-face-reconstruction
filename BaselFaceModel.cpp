#include "BaselFaceModel.h"

BaselFaceModel::BaselFaceModel(){
    setupFaces();
    setupLandmarkIndices();
    setupHDF5Data();
    initializeParameters();
    computeTransformationMatrix();
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

void BaselFaceModel::setupLandmarkIndices() {
    landmark_indices = FileReader::readIntFromTxt(LANDMARKS_FILE_PATH);
}

void BaselFaceModel::setupFaces() {
    faces = FileReader::readIntFromTxt(FACES_FILE_PATH);
}

void BaselFaceModel::setupHDF5Data() {
    shapeMean = FileReader::readHDF5File(HDF5_FILE_PATH, "/shape/model", "mean");
    shapePcaVariance = FileReader::readHDF5File(HDF5_FILE_PATH, "/shape/model", "pcaVariance");
    shapePcaBasis = FileReader::readMatrixHDF5File(HDF5_FILE_PATH, "/shape/model", "pcaBasis");

    expressionMean = FileReader::readHDF5File(HDF5_FILE_PATH, "/shape/expression", "mean");
    expressionPcaVariance = FileReader::readHDF5File(HDF5_FILE_PATH, "/shape/expression", "pcaVariance");
    expressionPcaBasis = FileReader::readMatrixHDF5File(HDF5_FILE_PATH, "/shape/expression", "pcaBasis");

    colorMean = FileReader::readHDF5File(HDF5_FILE_PATH, "/shape/model", "mean");
    colorPcaVariance = FileReader::readHDF5File(HDF5_FILE_PATH, "/shape/model", "pcaVariance");
    colorPcaBasis = FileReader::readMatrixHDF5File(HDF5_FILE_PATH, "/shape/model", "pcaBasis");
}

void BaselFaceModel::initializeParameters() {
    shapeParams = VectorXd::Zero(199);
    colorParams = VectorXd::Zero(199);
    expressionParams = VectorXd::Zero(100);
}

void BaselFaceModel::computeTransformationMatrix() {

}

std::vector<Vector3d> BaselFaceModel::getTransformedVertices() {
    return std::vector<Vector3d>();
}

std::vector<Vector3d> BaselFaceModel::getVerticesWithoutTransformation() {
    return std::vector<Vector3d>();
}

std::vector<Vector3d> BaselFaceModel::getNormals() {
    return std::vector<Vector3d>();
}

Vector3d BaselFaceModel::getColorValues() {
    return Eigen::Vector3d();
}

BaselFaceModel::~BaselFaceModel() = default;
