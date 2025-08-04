project("DeepSDF")
cmake_minimum_required(VERSION 3.0)

# Specify the path to Eigen (use local third-party version)
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third-party/eigen3.3.7")
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(CLI11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nanoflann REQUIRED)
# Use local Pangolin build instead of system-wide
set(PANGOLIN_DIR "${CMAKE_SOURCE_DIR}/third-party/Pangolin/build")
find_package(Pangolin REQUIRED PATHS ${PANGOLIN_DIR} NO_DEFAULT_PATH)
find_package(OpenGL REQUIRED)  # Include OpenGL

include_directories(${OPENGL_INCLUDE_DIRS})

# Add compiler definitions to ensure OpenGL functions are properly declared
add_definitions(-DGLEW_NO_GLU)

add_subdirectory(third-party/cnpy)

set(SOURCES
    src/ShaderProgram.cpp
    src/Utils.cpp)

add_executable(PreprocessMesh src/PreprocessMesh.cpp ${SOURCES})
target_include_directories(PreprocessMesh PRIVATE third-party/cnpy)
target_link_libraries(PreprocessMesh PRIVATE CLI11::CLI11 pango_core pango_display pango_geometry pango_glgeometry pango_opengl pango_image pango_windowing cnpy Eigen3::Eigen nanoflann::nanoflann OpenGL::GL)
target_compile_features(PreprocessMesh PRIVATE cxx_std_14)
set_target_properties(PreprocessMesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(SampleVisibleMeshSurface src/SampleVisibleMeshSurface.cpp ${SOURCES})
target_include_directories(SampleVisibleMeshSurface PRIVATE third-party/cnpy)
target_link_libraries(SampleVisibleMeshSurface PRIVATE CLI11::CLI11 pango_core pango_display pango_geometry pango_glgeometry pango_opengl pango_image pango_windowing cnpy Eigen3::Eigen nanoflann::nanoflann OpenGL::GL)
target_compile_features(SampleVisibleMeshSurface PRIVATE cxx_std_14)
set_target_properties(SampleVisibleMeshSurface PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")













