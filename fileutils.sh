#! /bin/sh
cp src/CMakeLists.fileutils.txt src/CMakeLists.txt
if test "$CUARCH" = ""; then export CUARCH=35; fi

# START
rm -rf _fileutils
mkdir _fileutils
cd _fileutils


# BUILD64
cmake -DM=64 -DCMAKE_CUDA_FLAGS="-arch=sm_$CUARCH" ../src
cmake --build . --config Debug


# END
cd ..