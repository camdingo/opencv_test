# CMakeList.txt : CMake project for opencv_test, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project(cvtest)

set( DCMAKE_TOOLCHAIN_FILE "D:/Code/vcpkg/Microsoft/vcpkg/scripts/buildsystems/vcpkg.cmake")

find_package(OpenCV)

set( CXX_SRCS
	opencv_gpu_test.cpp
)

include_directories( 
	"D:/Code/vcpkg/Microsoft/vcpkg/installed/x64-windows/include"
)
 
# Add source to this project's executable.
add_executable (opencv_test ${CXX_SRCS} ${OPEN_CV_LIBS} )

#target_link_libraries( opencv_test
#	"D:/Code/vcpkg/Microsoft/vcpkg/installed/x64-windows/lib/opencv_core.lib"
#	"D:/Code/vcpkg/Microsoft/vcpkg/installed/x64-windows/lib/opencv_highgui.lib"
#	"D:/Code/vcpkg/Microsoft/vcpkg/installed/x64-windows/lib/opencv_imgcodecs.lib"
#	"D:/Code/vcpkg/Microsoft/vcpkg/installed/x64-windows/lib/opencv_imgproc.lib"
#)
target_link_libraries( opencv_test
	opencv_core
	opencv_highgui
	opencv_imgproc
	opencv_imgcodecs
	opencv_cudaimgproc
	opencv_cudaarithm
)