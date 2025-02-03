#ifndef FACE_RECONSTRUCTION_FILEREADER_H
#define FACE_RECONSTRUCTION_FILEREADER_H

#include "Eigen.h"
#include <H5Cpp.h>

const std::string dataFolderPath = DATA_FOLDER_PATH;
const std::string resultFolderPath = RESULT_FOLDER_PATH;

class FileReader {
public:
    FileReader();
    ~FileReader();
    static std::vector<Vector3d> read3DVerticesFromTxt(const std::string& path);
    static std::vector<int> readIntFromTxt(const std::string& path);
    static std::vector<double> readHDF5File(const std::string& filePath, const std::string& groupPath, const std::string& datasetPath);
    static MatrixXd readMatrixHDF5File(const std::string& filePath, const std::string& groupPath, const std::string& datasetPath);
private:
};


#endif //FACE_RECONSTRUCTION_FILEREADER_H
