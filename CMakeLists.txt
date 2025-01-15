cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
set(CMAKE_SYSTEM_NAME Windows)
project(face_reconstruction)


# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(glog_DIR ${LIBRARY_DIR}/Glog/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4/ CACHE PATH "Path to Flann source folder")
set(OpenCV_DIR ${LIBRARY_DIR}/opencv/build/x64/vc16/lib CACHE PATH "Path to OpenCV folder")
set(HDF5_DIR ${LIBRARY_DIR}/HDF5/1.14.5/cmake CACHE PATH "Path to HDF5 folder")

add_subdirectory(${LIBRARY_DIR}/dlib/examples/../dlib dlib_build)

# Use pre-compiled libraries on Windows
#if(c
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
    set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage header file")
    set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage .lib/.dll folder")
endif()
#endif(WIN32)
set(HDF5_INCLUDE_DIR ${LIBRARY_DIR}/HDF5/1.14.5/include)
set(HDF5_LIBRARY_DIR ${LIBRARY_DIR}/HDF5/1.14.5/lib)

if(APPLE)
	set(FREE_IMAGE_DIR /usr/local/Cellar/freeimage/3.18.0) # set the lib path
	include_directories(${FREE_IMAGE_DIR}/include/) # include the lib
	link_directories(${FREE_IMAGE_DIR}/lib/) # link the lib
endif(APPLE)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

#find_package(nanoflann REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
#find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED)
get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${HDF5_INCLUDE_DIR})
link_directories(${HDF5_LIBRARY_DIR})
set(HDF5_LIBRARIES ${LIBRARY_DIR}/HDF5/1.14.5/lib/hdf5.lib)  # Adjust as per actual library filename (static or dynamic)
set(HDF5_CXX_LIBRARIES ${LIBRARY_DIR}/HDF5/1.14.5/lib/hdf5_cpp.lib)


add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(HEADER_FILES 
    Eigen.h 
    SimpleMesh.h 
    PointCloud.h 
    VirtualSensor.h 
    FreeImageHelper.h
    FacialLandmarks.h
    Optimization.h
	BFMParameters.h
)

set(SOURCE_FILES 
    FreeImageHelper.cpp
    Optimization.cpp
)

#set(GLOG_DLL_PATH_RELEASE "C:/Studium/3DScanning/Projekt/Face-Reconstruction/Libs/Glog/bin/libglogd.dll")

link_directories(${FreeImage_LIBRARY_DIR})
add_executable(face_reconstruction main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(face_reconstruction ceres freeimage Eigen3::Eigen dlib::dlib hdf5 hdf5_cpp)

target_include_directories(face_reconstruction PUBLIC ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR})
# On Windows copy dll to executable folder

    # Visual Studio properties
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT face_reconstruction)
set_property(TARGET face_reconstruction PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

# On Windows copy dll to executable folder
add_custom_command(TARGET face_reconstruction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${FreeImage_LIBRARY_DIR}/FreeImage.dll"
        ${CMAKE_BINARY_DIR})

add_custom_command(TARGET face_reconstruction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${GLOG_DLL_PATH_DEBUG}"
        ${CMAKE_BINARY_DIR})

#add_custom_command(TARGET face_reconstruction POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        "${GLOG_DLL_PATH_RELEASE}"
#        ${CMAKE_BINARY_DIR})

if(APPLE)
	find_library(LZ4_LIBRARY lz4 PATHS /usr/local/Cellar/lz4/1.10.0/lib REQUIRED)
	target_link_libraries(face_reconstruction ${LZ4_LIBRARY})
endif(APPLE)
