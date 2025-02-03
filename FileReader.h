#ifndef FACE_RECONSTRUCTION_FILEREADER_H
#define FACE_RECONSTRUCTION_FILEREADER_H

#include "Eigen.h"

const std::string dataFolderPath = DATA_FOLDER_PATH;
const std::string resultFolderPath = RESULT_FOLDER_PATH;

class FileReader {
public:
    FileReader();
    ~FileReader();
    static std::vector<Vector3d> read3DVerticesFromTxt(const std::string& path);
    static std::vector<int> readIntFromTxt(const std::string& path);
private:
};


#endif //FACE_RECONSTRUCTION_FILEREADER_H
