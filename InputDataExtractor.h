#ifndef FACE_RECONSTRUCTION_INPUTDATAEXTRACTOR_H
#define FACE_RECONSTRUCTION_INPUTDATAEXTRACTOR_H

#include <string>
#include "InputData.h"
#include <librealsense2/rs.hpp>

class InputDataExtractor {
public:
    InputDataExtractor();
    ~InputDataExtractor();
    static InputData extractInputData(const std::string& path);
};


#endif //FACE_RECONSTRUCTION_INPUTDATAEXTRACTOR_H
