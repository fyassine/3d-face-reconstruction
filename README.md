# 3DSMC-Face-Reconstruction
Members:

Fraj Yassine Lakhal

Luis Cardona Anaya 

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