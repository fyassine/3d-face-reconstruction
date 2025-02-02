#include "BaselFaceModel.h"

BaselFaceModel::BaselFaceModel(){
    setupFaces();
    setupLandmarkIndices();
    setupHDF5Data();
    initializeParameters();
    //TODO read hdf5
}

BaselFaceModel::~BaselFaceModel() = default;
