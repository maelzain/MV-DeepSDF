#!/bin/bash -e
#
echo "Installing Eigen3 ..."
# git_clone "git clone --branch=3.4.0 --depth=1 https://gitlab.com/libeigen/eigen.git"
cd eigen
if [ ! -d build ]; then
  mkdir build
fi
if [ ! -d install ]; then
  mkdir install
fi
cd build
cmake -DCMAKE_INSTALL_PREFIX="$(pwd)/../install" ..
make -j8
make install
cd ../..
  
  
echo "Installing Pangolin ..."
# git_clone "git clone --recursive --depth=1 https://github.com/stevenlovegrove/Pangolin.git"
cd Pangolin
if [ ! -d build ]; then
  mkdir build
fi
cd build
cmake ..
make -j8
Pangolin_DIR=$(pwd)
cd ../..
  
echo "building DeepSDF ..."
if [ ! -d build ]; then
  mkdir build
fi
cd build
cmake \
  -DEigen3_DIR="$(pwd)/../third-party/eigen/install/share/eigen3/cmake" \
  -DPangolin_DIR="$(pwd)/../third-party/Pangolin/build" \
  ..
make -j8
