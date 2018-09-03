#! /bin/sh
cp src/CMakeLists.full.txt src/CMakeLists.txt

# START
rm -rf _build
mkdir _build
cd _build


# BUILD64
mkdir build64
cd build64
cmake -DM=64 -DC64=ON -DCMAKE_CUDA_FLAGS="-arch=sm_60" ../../src
cd ..
cmake --build build64 --config Debug
cd build64
make test
cd ..


# BUILD32
mkdir build32
cd build32
#cmake -DM=32 -DC64=OFF -DCMAKE_CUDA_FLAGS="-arch=sm_60" ../../src
cd ..

#cmake --build build32 --config Debug
cd build32
#make test
cd ..


# END
cd ..