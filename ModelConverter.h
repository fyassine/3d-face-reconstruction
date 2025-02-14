#ifndef FACE_RECONSTRUCTION_MODELCONVERTER_H
#define FACE_RECONSTRUCTION_MODELCONVERTER_H

#include "Eigen.h"
#include "InputDataExtractor.h"
#include "BaselFaceModel.h"
#include "InputData.h"

class ModelConverter {
public:
    ModelConverter();
    ~ModelConverter();
    static void convertToPly(std::vector<Vector3d> vertices, const std::string& path);
    static void convertToPly(std::vector<Vector3d> vertices, const std::vector<Vector3i>& color, const std::vector<int>& faces, const std::string& path);
    static void convertImageToPly(const std::vector<double>& depth, const std::vector<Eigen::Vector3d>& colorValues, const std::string& path,
                                  const Matrix3d& intrinsics, const Matrix4d& extrinsics);
    static void generateGeometricErrorModel(BaselFaceModel* bfm, InputData* inputData, const std::string& path);
    static void convertBFMToPly(BaselFaceModel* bfm, const std::string& path);
private:
};


#endif //FACE_RECONSTRUCTION_MODELCONVERTER_H
