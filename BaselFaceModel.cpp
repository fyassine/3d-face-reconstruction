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

const std::vector<Vector3d> &BaselFaceModel::getNormals() const {
    return normals;
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
    for (const auto & v : vertices) {
        Vector4d oldVertex(v.x(), v.y(), v.z(), 1.0);
        Vector4d newVertex = transformation * oldVertex;
        transformedVertices.emplace_back(newVertex.x(), newVertex.y(), newVertex.z());
    }
    return transformedVertices;
}

void BaselFaceModel::updateNormals() {
    // Get the list of vertices (assuming getVerticesWithoutTransformation gives us the vertices)
    const std::vector<Vector3d>& vertices = getVerticesWithoutTransformation();

    // Vector to store the normals for each vertex
    std::vector<Vector3d> newNormals(vertices.size(), Vector3d(0.0, 0.0, 0.0));

    // Iterate over faces to compute the normals
    for (size_t i = 0; i < faces.size(); i += 3) {
        // Fetch the indices of the three vertices of the face
        int v0_index = faces[i];
        int v1_index = faces[i + 1];
        int v2_index = faces[i + 2];

        // Get the positions of the vertices of the triangle
        const Vector3d& v0 = vertices[v0_index];
        const Vector3d& v1 = vertices[v1_index];
        const Vector3d& v2 = vertices[v2_index];

        // Compute the two edge vectors
        Vector3d edge1 = v1 - v0;
        Vector3d edge2 = v2 - v0;

        // Compute the normal for the face using the cross product
        Vector3d faceNormal = edge1.cross(edge2).normalized();

        // Accumulate the normals for each vertex
        newNormals[v0_index] += faceNormal;
        newNormals[v1_index] += faceNormal;
        newNormals[v2_index] += faceNormal;
    }

    // Normalize the normals for each vertex
    for (auto& normal : newNormals) {
        normal.normalize();
    }
    normals = newNormals;
}

