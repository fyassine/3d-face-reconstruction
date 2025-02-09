#ifndef FACE_RECONSTRUCTION_BASELFACEMODEL_H
#define FACE_RECONSTRUCTION_BASELFACEMODEL_H

#include "Eigen.h"
#include "ProcrustesAligner.h"
#include "FileReader.h"
#include "InputData.h"

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
    void computeTransformationMatrix(InputData* inputData);

    std::vector<Vector3d> getVerticesWithoutTransformation();
    std::vector<Vector3d> transformVertices(const std::vector<Vector3d>& vertices);
    std::vector<Vector3i> getColorValues();

    std::vector<Vector3d> getLandmarks();
    static std::vector<Vector3d> getNormals(); //TODO

    [[nodiscard]] const std::vector<double> &getColorMean() const;
    [[nodiscard]] const MatrixXd &getColorPcaBasis() const;
    [[nodiscard]] const std::vector<double> &getColorPcaVariance() const;
    [[nodiscard]] const std::vector<double> &getShapeMean() const;
    [[nodiscard]] const MatrixXd &getShapePcaBasis() const;
    [[nodiscard]] const std::vector<double> &getShapePcaVariance() const;
    [[nodiscard]] const std::vector<double> &getExpressionMean() const;
    [[nodiscard]] const MatrixXd &getExpressionPcaBasis() const;
    [[nodiscard]] const std::vector<double> &getExpressionPcaVariance() const;
    VectorXd &getColorParams();
    VectorXd &getShapeParams();
    VectorXd &getExpressionParams();
    [[nodiscard]] const Matrix4d &getTransformation() const;
    [[nodiscard]] const std::vector<int> &getLandmarkIndices() const;
    [[nodiscard]] const std::vector<int> &getFaces() const;

private:
    Matrix4d transformation;
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

    VectorXd colorParams;
    VectorXd shapeParams;
    VectorXd expressionParams;
};


#endif //FACE_RECONSTRUCTION_BASELFACEMODEL_H
