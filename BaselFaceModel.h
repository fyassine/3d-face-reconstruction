#ifndef FACE_RECONSTRUCTION_BASELFACEMODEL_H
#define FACE_RECONSTRUCTION_BASELFACEMODEL_H

#include "Eigen.h"
#include "ProcrustesAligner.h"
#include "FileReader.h"

#define HDF5_FILE_PATH "model2019_face12.h5"
#define FACES_FILE_PATH "faces.txt"
#define LANDMARKS_FILE_PATH "landmarks.txt"

class BaselFaceModel {
public:
    BaselFaceModel();
    ~BaselFaceModel();

    void setupLandmarkIndices();
    void setupFaces();
    void setupHDF5Data();
    void initializeParameters();
    void computeTransformationMatrix();

    std::vector<Vector3d> getTransformedVertices();
    std::vector<Vector3d> getVerticesWithoutTransformation();
    Vector3d getVertex(int vertexId);

    std::vector<Vector3d> getLandmarks();
    std::vector<Vector3d> getNormals();

    std::vector<Vector3d> transformVertices(std::vector<Vector3d> vertices);

private:

    Matrix4f transformation; //TODO: really here?!
    std::vector<int> landmark_indices;
    std::vector<int> faces;

    std::vector<double> colorMean;
    MatrixXd colorPcaBasis;
    std::vector<double> colorPcaVariance;

    std::vector<double> shapeMean;
    MatrixXd shapePcaBasis;
    std::vector<double> shapePcaVariance;

    std::vector<double> expressionMean;
    MatrixXd expressionPcaBasis;
    std::vector<double> expressionPcaVariance;

    Eigen::VectorXd colorParams;
    Eigen::VectorXd shapeParams;
    Eigen::VectorXd expressionParams;
};


#endif //FACE_RECONSTRUCTION_BASELFACEMODEL_H
