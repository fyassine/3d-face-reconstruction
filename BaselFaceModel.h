#ifndef FACE_RECONSTRUCTION_BASELFACEMODEL_H
#define FACE_RECONSTRUCTION_BASELFACEMODEL_H

#include "Eigen.h"
const std::string dataFolderPath = DATA_FOLDER_PATH;
const std::string resultFolderPath = RESULT_FOLDER_PATH;

class BaselFaceModel {
public:
    BaselFaceModel();
    ~BaselFaceModel();

    void setupLandmarkIndices();
    void setupFaces();
    void setupHDF5Data();
    void initializeParameters();

    std::vector<Vector3d> getAllVertices();
    Vector3d getVertex(int vertexId);

    std::vector<Vector3d> getNormals();

private:
    Matrix4f transformation; //TODO: really here?!
    std::vector<int> landmark_indices;
    std::vector<int> faces;

    std::vector<double> colorMean;
    std::vector<double> colorPcaBasis;
    std::vector<double> colorPcaVariance;

    std::vector<double> shapeMean;
    std::vector<double> shapePcaBasis;
    std::vector<double> shapePcaVariance;

    std::vector<double> expressionMean;
    std::vector<double> expressionPcaVariance;
    std::vector<double> expressionPcaBasis;

    Eigen::VectorXd colorParams;
    Eigen::VectorXd shapeParams;
    Eigen::VectorXd expressionParams;
};


#endif //FACE_RECONSTRUCTION_BASELFACEMODEL_H
