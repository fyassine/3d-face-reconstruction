#ifndef FACE_RECONSTRUCTION_BASELFACEMODEL_H
#define FACE_RECONSTRUCTION_BASELFACEMODEL_H

#include "Eigen.h"
#include "ProcrustesAligner.h"

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
