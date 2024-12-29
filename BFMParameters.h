#ifndef FACE_RECONSTRUCTION_BFMPARAMETERS_H
#define FACE_RECONSTRUCTION_BFMPARAMETERS_H

#include <iostream>
#include <H5Cpp.h>

void printStructure(const H5::Group& group, const std::string& prefix = "") {
    for (hsize_t i = 0; i < group.getNumObjs(); ++i) {
        std::string name = group.getObjnameByIdx(i);

        // Object information structure
        H5O_info2_t obj_info;

        // Correct call to H5Oget_info_by_name3
        H5Oget_info_by_name3(group.getId(), name.c_str(), &obj_info, H5O_INFO_BASIC, H5P_DEFAULT);

        // Check object type and recurse if it's a group
        if (obj_info.type == H5O_TYPE_GROUP) {
            std::cout << prefix << "Group: " << name << std::endl;
            H5::Group subgroup = group.openGroup(name);
            printStructure(subgroup, prefix + "  ");
        } else if (obj_info.type == H5O_TYPE_DATASET) {
            std::cout << prefix << "Dataset: " << name << std::endl;
        }
    }
}

void readH5File(const std::string& filename){
    H5::H5File file(filename, H5F_ACC_RDONLY);
    if (!file.getId()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    std::cout << "File opened successfully!" << std::endl;
    printStructure(file.openGroup("/"));
    std::cout << "File size: " << file.getFileSize() << std::endl;
}



#endif //FACE_RECONSTRUCTION_BFMPARAMETERS_H
