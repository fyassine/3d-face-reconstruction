# 3DSMC-Face-Reconstruction
Members:

Fraj Yassine Lakhal

Neli Shahapuni

Leo Keber

Face reconstruction project for 3DSMC Winter 24/25.

Add the following to you CMakelists.txt
```cmake
set(DATA_FOLDER_PATH "../../Data/")
add_definitions(-DDATA_FOLDER_PATH=\"${DATA_FOLDER_PATH}\")

set(RESULT_FOLDER_PATH "../../Result/")
add_definitions(-DRESULT_FOLDER_PATH=\"${RESULT_FOLDER_PATH}\")
```

You have to compile cereal under Libs/
Inside CMake just add the following:
```cmake
include_directories(${LIBRARY_DIR}/cereal/include)
```

File Structure:
```
root/
├── 3DSMC-Face-Reconstruction/
├── Data/
└── Result/
    ├── Expression_Transfer_Video/
    ├── Expression_Frames_Reconstructed/
    ├── Source_Frames_Reconstructed/
    ├── Source_Frames/
    ├── Target_Frames/
```
