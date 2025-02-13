cmake_minimum_required(VERSION 3.6)

cmake_policy(SET CMP0074 OLD)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)
set(CMAKE_SYSTEM_NAME Windows)
project(face_reconstruction)

set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(DEBUG_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug-visual-studio/Debug)
set(VCPKG_ROOT C:/VCPKG/vcpkg)

set(DATA_FOLDER_PATH "../../../Data/")
add_definitions(-DDATA_FOLDER_PATH=\"${DATA_FOLDER_PATH}\")
set(RESULT_FOLDER_PATH "../../../Result/")
add_definitions(-DRESULT_FOLDER_PATH=\"${RESULT_FOLDER_PATH}\")

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(flann REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(FreeImage REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(Stb REQUIRED)
find_package(realsense2 CONFIG REQUIRED)
find_package(cereal CONFIG REQUIRED)


add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

set(HEADER_FILES
		Eigen.h
		FreeImageHelper.h
		FacialLandmarks.h
		Renderer.h
		BaselFaceModel.h
		FileReader.h
		Optimizer.h
		SingleInputFrame.h
		InputData.h
		InputDataExtractor.h
		ModelConverter.h
		FaceReconstructor.h
)

set(SOURCE_FILES
		FreeImageHelper.cpp
		Renderer.cpp
		BaselFaceModel.cpp
		FileReader.cpp
		Optimizer.cpp
		SingleInputFrame.cpp
		InputData.cpp
		InputDataExtractor.cpp
		ModelConverter.cpp
		FaceReconstructor.cpp
)

add_executable(face_reconstruction main.cpp ${HEADER_FILES} ${SOURCE_FILES})

set(Stb_INCLUDE_DIR {VCPKG_ROOT}/packages/stb_x64-windows/include)
target_include_directories(face_reconstruction PRIVATE
		${Stb_INCLUDE_DIR}
)

target_link_libraries(face_reconstruction PRIVATE
		ceres
		freeimage::FreeImage
		Eigen3::Eigen
		dlib::dlib
		ZLIB::ZLIB
		HDF5::HDF5
		opencv_core
		opencv_imgproc
		opencv_highgui
		opencv_imgcodecs
		opencv_videoio
		opencv_calib3d  # Add this to link with the calib3d module
		glfw
		GLEW::GLEW
		realsense2::fw realsense2::rsutils realsense2::realsense2 realsense2::realsense-file
		cereal::cereal
)

add_custom_command(TARGET face_reconstruction POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${VCPKG_ROOT}/installed/x64-windows/bin/hdf5_cpp.dll
		${DEBUG_BUILD_DIR}
)

# Visual Studio properties
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT face_reconstruction)
set_property(TARGET face_reconstruction PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)