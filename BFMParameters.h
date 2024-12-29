#ifndef FACE_RECONSTRUCTION_BFMPARAMETERS_H
#define FACE_RECONSTRUCTION_BFMPARAMETERS_H

#include <iostream>
#include <H5Cpp.h>

void readH5File(const std::string& filename){
    H5::H5File file(filename, H5F_ACC_RDONLY);
    /*if (!file.getId()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }*/
    std::cout << "File opened successfully!" << std::endl;
    //std::cout << "File size: " << file.getFileSize() << std::endl;
}



#endif //FACE_RECONSTRUCTION_BFMPARAMETERS_H
