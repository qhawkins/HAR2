cmake_minimum_required(VERSION 3.10) # Set the minimum version of CMake required

project(HAR2) # Name of your project
# Set the C++ standard you wish to use (you could use C++11, C++14, C++17, etc.)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE CUDA_STANDARD 12.2)


#find_package(Boost 1.84.0 REQUIRED COMPONENTS json)
#find_package(Boost COMPONENTS system REQUIRED) # Find Boost system library

#find_package(OpenSSL REQUIRED)
# Add executable target with source files
add_executable(HAR2 main.cpp)
include_directories("~/Desktop/Eigen/eigen/Eigen")
include_directories("/usr/local/include")
SET(NLOPT_LIBRARY "/usr/local/lib/libnlopt.so")
SET(NLOPT_LIBRARIES "/usr/local/lib/libnlopt.so.0" "/usr/local/lib/libnlopt.so.0.11.1")

#add_executable(CudaNse bs.cu)
#set_target_properties(CudaNse PROPERTIES LINKER_LANGUAGE CUDA CUDA_ARCHITECTURES 89)
target_link_libraries(HAR2 ${NLOPT_LIBRARY} ${NLOPT_LIBRARIES} ${CMAKE_DL_LIBS} pthread)

#target_link_libraries(NSEDataCollection Boost::system ssl crypto ${CMAKE_DL_LIBS} pthread)
#target_link_libraries(NSEDataCollection Boost::json Boost::system OpenSSL::SSL OpenSSL::Crypto ${CMAKE_DL_LIBS} pthread)

#target_link_options(NSEDataCollection PRIVATE -lssl -lcrypto)

# If you have include directories you wish to add, you can use include_directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# If you have libraries to link against, use target_link_libraries
# target_link_libraries(MyExecutable myLibrary)
