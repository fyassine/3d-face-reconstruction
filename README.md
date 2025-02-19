# 3DSMC-Face-Reconstruction
Members:

Fraj Yassine Lakhal

Neli Shahapuni

Leo Keber

## How to setup the project on your local machine:

Face reconstruction project for 3DSMC Winter 24/25.

Add the following to you CMakelists.txt
```cmake
set(DATA_FOLDER_PATH "../../Data/")
add_definitions(-DDATA_FOLDER_PATH=\"${DATA_FOLDER_PATH}\")

set(RESULT_FOLDER_PATH "../../Result/")
add_definitions(-DRESULT_FOLDER_PATH=\"${RESULT_FOLDER_PATH}\")
```

Ensure you include the following libraries in your Cmake project (configuration may vary):
- Eigen
- flann
- Ceres
- glog
- OpenCV
- dlib
- ZLIB
- HDF5
- FreeImage REQUIRED
- Stb REQUIRED
- realsense2
- cereal

Optional (should not be needed now, but we had older code which relied on OpenGL):
- glfw3
- GLEW

File Structure:
```
root/
├── 3DSMC-Face-Reconstruction/ // <- this folder includes the git repository
├── Data/ // <- input data
└── Result/
    ├── Source_Backprojections // <- depth data from Intel RealSense
    ├── Source_GeometricError_Procrustes/ // <- errors (procrustes, sparse, dense)
    ├── Source_GeometricError_Sparse/ 
    ├── Source_GeometricError_Dense/
    ├── Source_Models_TextureMap/ // <- Model with texture map
    ├── Source_Models_Procrustes/ // <- Model after running Procrustes
    ├── Source_Models_Sparse/ // <- Model after sparse optimization
    ├── Source_Models_Dense/ // <- Model after dense optimization
    ├── Source_Frames/ // <- All frames
    ├── Source_Frames_Reconstructed/ // <- All frames after reconstruction
    ├── Source_Frames_Reconstructed_Texture/ // <- All frames after reconstruction with texture
```
Use this link to get the .bag-files/input_data for your Data folder:
https://syncandshare.lrz.de/getlink/fi7jFFfLVfxm4nhPFeisqN/
