@echo off
copy /Y src\CMakeLists.fileutils.txt src\CMakeLists.txt > nul
IF "%CUARCH%"=="" SET CUARCH=35

rem START
rd /s /q _fileutils
rem rm -rf _fileutils
mkdir _fileutils
pushd _fileutils


rem BUILD64
cmake -G "Visual Studio 15 2017 Win64" -Darch="%CUARCH%" -DCMAKE_CUDA_FLAGS="-arch=sm_%CUARCH%" ../src
cmake --build . --config Debug


rem END
popd