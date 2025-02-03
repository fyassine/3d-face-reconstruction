#ifndef FACE_RECONSTRUCTION_MODELCONVERTER_H
#define FACE_RECONSTRUCTION_MODELCONVERTER_H

#include "Eigen.h"

class ModelConverter {
public:
    ModelConverter();
    ~ModelConverter();
    static void convertToPly(std::vector<Vector3d> vertices, const std::vector<Vector3i>& color, const std::vector<int>& faces, const std::string& path);
private:
};


#endif //FACE_RECONSTRUCTION_MODELCONVERTER_H
