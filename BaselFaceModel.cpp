#include "BaselFaceModel.h"

BaselFaceModel::BaselFaceModel(){
    setupFaces();
    setupLandmarkIndices();
    setupHDF5Data();
    initializeParameters();
}

BaselFaceModel::~BaselFaceModel() = default;

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

    expressionMean = FileReader::readHDF5File(HDF5_FILE_PATH, "/expression/model", "mean");
    expressionPcaVariance = FileReader::readHDF5File(HDF5_FILE_PATH, "/expression/model", "pcaVariance");
    expressionPcaBasis = FileReader::readMatrixHDF5File(HDF5_FILE_PATH, "/expression/model", "pcaBasis");

    colorMean = FileReader::readHDF5File(HDF5_FILE_PATH, "/color/model", "mean");
    colorPcaVariance = FileReader::readHDF5File(HDF5_FILE_PATH, "/color/model", "pcaVariance");
    colorPcaBasis = FileReader::readMatrixHDF5File(HDF5_FILE_PATH, "/color/model", "pcaBasis");
    std::cout << shapeMean[0] << std::endl;
}

void BaselFaceModel::initializeParameters() {
    shapeParams = VectorXd::Zero(199);
    colorParams = VectorXd::Zero(199);
    expressionParams = VectorXd::Zero(100);
}

void BaselFaceModel::computeTransformationMatrix(InputData* inputData) {
    ProcrustesAligner aligner;
    auto landmarks_bfm = getLandmarks();
    auto landmarks_image = inputData->getMCurrentFrame().getMLandmarks();

    std::vector<Vector3d> source(landmarks_bfm.begin() + 18, landmarks_bfm.end());
    std::vector<Vector3d> target(landmarks_image.begin() + 18, landmarks_image.end());

    transformation = aligner.estimatePose(source, target);
}

std::vector<Vector3d> BaselFaceModel::getTransformedVertices() {
    return std::vector<Vector3d>();
}

std::vector<Vector3d> BaselFaceModel::getVerticesWithoutTransformation() {

    std::vector<Vector3d> vertices;

    Eigen::VectorXd shapeVar = Eigen::Map<Eigen::VectorXd>(shapePcaVariance.data(), (int) shapePcaVariance.size());
    Eigen::VectorXd modifiedShape = shapePcaBasis * (shapeVar.cwiseSqrt().cwiseProduct(shapeParams));
    Eigen::VectorXd expressionVar = Eigen::Map<Eigen::VectorXd>(expressionPcaVariance.data(), (int) expressionPcaVariance.size());
    Eigen::VectorXd modifiedExpression = expressionPcaBasis * (expressionVar.cwiseSqrt().cwiseProduct(expressionParams));

    int n = (int) shapeMean.size();
    for (int i = 0; i < n; i+=3) {
        Eigen::Vector3d newVertex;
        newVertex.x() = shapeMean[i] + expressionMean[i] + modifiedShape[i] + modifiedExpression[i];
        newVertex.y() = shapeMean[i + 1] + expressionMean[i + 1] + modifiedShape[i + 1] + modifiedExpression[i + 1];
        newVertex.z() = shapeMean[i + 2] + expressionMean[i + 2] + modifiedShape[i + 2] + modifiedExpression[i + 2];
        vertices.emplace_back(newVertex);
    }
    return vertices;
}

std::vector<Vector3i> BaselFaceModel::getColorValues() {
    Eigen::VectorXd colorVar = Eigen::Map<Eigen::VectorXd>(colorPcaVariance.data(), (int) colorPcaVariance.size());
    Eigen::VectorXd modifiedColor = colorPcaBasis * (colorVar.cwiseSqrt().cwiseProduct(colorParams));
    std::vector<Eigen::Vector3i> colorValues;

    std::cout << colorPcaBasis(0, 0) << std::endl;

    int n = (int) shapeMean.size();
    for (int i = 0; i < n; i+=3) {
        Eigen::Vector3i newColorValue;
        newColorValue.x() = (int) ((colorMean[i] + modifiedColor[i]) * 255);
        newColorValue.y() = (int) ((colorMean[i + 1] + modifiedColor[i + 1]) * 255);
        newColorValue.z() = (int) ((colorMean[i + 2] + modifiedColor[i + 2]) * 255);

        if(newColorValue.x() < 0){
            newColorValue.x() = 0;
        }else if(newColorValue.x() > 255){
            newColorValue.x() = 255;
        }

        if(newColorValue.y() < 0){
            newColorValue.y() = 0;
        }else if(newColorValue.y() > 255){
            newColorValue.y() = 255;
        }

        if(newColorValue.z() < 0){
            newColorValue.z() = 0;
        }else if(newColorValue.z() > 255){
            newColorValue.z() = 255;
        }

        colorValues.emplace_back(newColorValue);
    }
    return colorValues;
}

std::vector<Vector3d> BaselFaceModel::getNormals() {
    return std::vector<Vector3d>();
}

const std::vector<double> &BaselFaceModel::getColorMean() const {
    return colorMean;
}

const MatrixXd &BaselFaceModel::getColorPcaBasis() const {
    return colorPcaBasis;
}

const std::vector<double> &BaselFaceModel::getColorPcaVariance() const {
    return colorPcaVariance;
}

const std::vector<double> &BaselFaceModel::getShapeMean() const {
    return shapeMean;
}

const MatrixXd &BaselFaceModel::getShapePcaBasis() const {
    return shapePcaBasis;
}

const std::vector<double> &BaselFaceModel::getShapePcaVariance() const {
    return shapePcaVariance;
}

const std::vector<double> &BaselFaceModel::getExpressionMean() const {
    return expressionMean;
}

const MatrixXd &BaselFaceModel::getExpressionPcaBasis() const {
    return expressionPcaBasis;
}

const std::vector<double> &BaselFaceModel::getExpressionPcaVariance() const {
    return expressionPcaVariance;
}

VectorXd &BaselFaceModel::getColorParams() {
    return colorParams;
}

VectorXd &BaselFaceModel::getShapeParams() {
    return shapeParams;
}

VectorXd &BaselFaceModel::getExpressionParams() {
    return expressionParams;
}

const Matrix4d &BaselFaceModel::getTransformation() const {
    return transformation;
}

const std::vector<int> &BaselFaceModel::getLandmarkIndices() const {
    return landmark_indices;
}

const std::vector<int> &BaselFaceModel::getFaces() const {
    return faces;
}

std::vector<Vector3d> BaselFaceModel::transformVertices(const std::vector<Vector3d>& vertices) {
    std::vector<Vector3d> transformedVertices;
    for (int i = 0; i < vertices.size(); ++i) {
        Vector4d oldVertex(vertices[i].x(), vertices[i].y(), vertices[i].z(), 1.0);
        Vector4d newVertex = transformation * oldVertex;
        transformedVertices.emplace_back(newVertex.x(), newVertex.y(), newVertex.z());
    }
    return transformedVertices;
}

void BaselFaceModel::expressionTransfer(BaselFaceModel *baselFaceModel) {
    expressionParams = baselFaceModel->getExpressionParams();
}
