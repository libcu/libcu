# https://devblogs.nvidia.com/parallelforall/building-cuda-applications-cmake
# https://cmake.org/cmake/help/v3.9/manual/cmake-buildsystem.7.html
# http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(libcu LANGUAGES CXX CUDA)
set(COMPILE_FLAGSO2 "-Xcicc -O0 -Xcudafe \"--diag_suppress=divide_by_zero\"")
include(CTest)
enable_testing()

include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# CORE
add_library(libcu.${arch} STATIC
  libcu/sentinel-msg.cpp
  libcu/sentinel.cpp
  libcu/cuda_runtime.cpp
  libcu/stringcu.cu
  libcu/libcu.cu
  libcu/libcu.stdlib.cu

  #libcu/crtdefscu.cu
  #libcu/ctypecu.cu
  #libcu/errnocu.cu
  #libcu/regexcu.cu
  #libcu/sentinel-gpu.cu
  #libcu/setjmpcu.cu
  #libcu/stdiocu.cu
  #libcu/stdlibcu.cu
  #libcu/timecu.cu
  #libcu/unistdcu.cu
  #libcu/direntcu.cu
  #libcu/sys/statcu.cu
  #libcu/sys/timecu.cu
  #libcu/ext/hash.cu
  #libcu/ext/memfile.cu
  #libcu/ext/mutex.cu
  #libcu/grpcu.cu
  #libcu/pwdcu.cu
  #libcu/fsystem.cu
  #libcu/fcntlcu.cu
  )
  set_source_files_properties(libcu/stringcu.cu libcu/libcu.cu PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")

add_library(libcu.fileutils.${arch} STATIC
  libcu.fileutils/libcu.fileutils.cu
  libcu.fileutils/sentinel-msg.cpp
  )
target_link_libraries(libcu.fileutils.${arch} PRIVATE libcu.${arch})

# Request that libcu be built with -std=c++11. As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(libcu.${arch} PUBLIC cxx_std_11)
target_compile_features(libcu.fileutils.${arch} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle library to be built with -dc as the member functions could be called by other libraries and executables
set_target_properties(libcu.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(libcu.fileutils.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
if (M)
  set_target_properties(libcu.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(libcu.fileutils.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
endif()


# TCL
add_library(libcu.jimtcl.${arch} STATIC
  tcl/libcu.jimtcl/jim.cu
  tcl/libcu.jimtcl/libcu.jimtcl.cu

  #tcl/libcu.jimtcl/jim-interactive.cu
  #tcl/libcu.jimtcl/init.cu
  #tcl/libcu.jimtcl/jim-aio.cu
  #tcl/libcu.jimtcl/jim-array.cu
  #tcl/libcu.jimtcl/jim-clock.cu
  #tcl/libcu.jimtcl/jim-eventloop.cu
  #tcl/libcu.jimtcl/jim-exec.cu
  #tcl/libcu.jimtcl/jim-file.cu
  #tcl/libcu.jimtcl/jim-format.cu
  #tcl/libcu.jimtcl/jim-gpu.cu
  #tcl/libcu.jimtcl/jim-history.cu
  #tcl/libcu.jimtcl/jim-load.cu
  #tcl/libcu.jimtcl/jim-namespace.cu
  #tcl/libcu.jimtcl/jim-pack.cu
  #tcl/libcu.jimtcl/jim-package.cu
  #tcl/libcu.jimtcl/jim-readdir.cu
  #tcl/libcu.jimtcl/jim-regexp.cu
  #tcl/libcu.jimtcl/jim-subcmd.cu
  #_tcl/libcu.jimtcl/jim-tclprefix.cu
  #tcl/libcu.jimtcl/jimex.cu
  #tcl/libcu.jimtcl/utf8.cu
  )
  set_source_files_properties(tcl/libcu.jimtcl/jim.cu tcl/libcu.jimtcl/libcu.jimtcl.cu PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGSO2}")
  target_compile_definitions(libcu.jimtcl.${arch} PRIVATE jim_ext_aio jim_ext_namespace jim_ext_package JIM_MAINTAINER JIM_REFERENCES)
  target_link_libraries(libcu.jimtcl.${arch} PRIVATE libcu.${arch})

include_directories(tcl/libcu.jimtcl)
add_executable(jimtcl.${arch}
	tcl/jimtcl/jimsh.cu
  )
  target_link_libraries(jimtcl.${arch} PRIVATE libcu.jimtcl.${arch})

add_library(libcu.tinytcl.${arch} STATIC
  tcl/libcu.tinytcl/libcu.tinytcl.cu

  #tcl/libcu.tinytcl/tclAlloc.cu
  #tcl/libcu.tinytcl/tclAssem.cu
  #tcl/libcu.tinytcl/tclBasic.cu
  #tcl/libcu.tinytcl/tclCmdAH.cu
  #tcl/libcu.tinytcl/tclCmdIL.cu
  #tcl/libcu.tinytcl/tclCmdMZ.cu
  #tcl/libcu.tinytcl/tclExpr.cu
  #tcl/libcu.tinytcl/tclGet.cu
  #tcl/libcu.tinytcl/tclGlob.cu
  #tcl/libcu.tinytcl/tclGpuAZ.cu
  #tcl/libcu.tinytcl/tclGpuEnv.cu
  #tcl/libcu.tinytcl/tclGpuStr.cu
  #tcl/libcu.tinytcl/tclGpuUtil.cu
  #tcl/libcu.tinytcl/tclHash.cu
  #tcl/libcu.tinytcl/tclHistory.cu
  #tcl/libcu.tinytcl/tclLoad-noop.cu
  #tcl/libcu.tinytcl/tclLoad.cu
  #tcl/libcu.tinytcl/tclObj.cu
  #tcl/libcu.tinytcl/tclParse.cu
  #tcl/libcu.tinytcl/tclProc.cu
  #tcl/libcu.tinytcl/tclUtil.cu
  #tcl/libcu.tinytcl/tclVar.cu
  #tcl/libcu.tinytcl/tclExDbg.cu
  #tcl/libcu.tinytcl/tclNoExtensions.cu
  #tcl/libcu.tinytcl/tclExGen.cu
  #tcl/libcu.tinytcl/tclMod.cu
  )
  set_source_files_properties(tcl/libcu.tinytcl/libcu.tinytcl.cu PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGSO2}")
  target_compile_definitions(libcu.tinytcl.${arch} PRIVATE TCL_MEM_DEBUG)
  target_link_libraries(libcu.tinytcl.${arch} PRIVATE libcu.${arch})

include_directories(tcl/libcu.tinytcl)
add_executable(tinytcl.${arch}
	tcl/tinytcl/tinySh.cu
  )
  target_link_libraries(tinytcl.${arch} PRIVATE libcu.tinytcl.${arch})

# Request that libcu be built with -std=c++11. As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(libcu.jimtcl.${arch} PUBLIC cxx_std_11)
target_compile_features(jimtcl.${arch} PUBLIC cxx_std_11)
target_compile_features(libcu.tinytcl.${arch} PUBLIC cxx_std_11)
target_compile_features(tinytcl.${arch} PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle library to be built with -dc as the member functions could be called by other libraries and executables
set_target_properties(libcu.jimtcl.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(jimtcl.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(libcu.tinytcl.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tinytcl.${arch} PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
if (M)
  set_target_properties(libcu.jimtcl.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(jimtcl.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(libcu.tinytcl.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
  set_target_properties(tinytcl.${arch} PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64})
endif()


# TESTS
if (BUILD_TESTING)
  add_executable(libcu_tests
	libcu.tests/libcu.tests.cu
	libcu.tests/program.cu

  #libcu.tests/crtdefsTest.cu
  #libcu.tests/ctypeTest.cu
  #libcu.tests/direntTest.cu
  #libcu.tests/errnoTest.cu
  #libcu.tests/fcntlTest.cu
  #libcu.tests/fsystemTest.cu
  #libcu.tests/grpTest.cu
  #libcu.tests/hostTest.cu
  #libcu.tests/pwdTest.cu
  #libcu.tests/regexTest.cu
  #libcu.tests/sentinelTest.cu
  #libcu.tests/setjmpTest.cu
  #libcu.tests/stddefTest.cu
  #libcu.tests/stdioTest.cu
  #libcu.tests/stdlibTest.cu
  #libcu.tests/stringTest.cu
  #libcu.tests/timeTest.cu
  #libcu.tests/unistdTest.cu
  )
  set_source_files_properties(libcu.tests/libcu.tests.cu PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")

  set_target_properties(libcu_tests PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON)
  if (M)
    set_target_properties(libcu_tests PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}" CUDA_64_BIT_DEVICE_CODE ${C64} CUDA_SEPARABLE_COMPILATION ON)
  endif()
  target_link_libraries(libcu_tests PRIVATE libcu.fileutils.${arch})

  add_test(NAME crtdefs_test1 COMMAND libcu_tests 1)
  add_test(NAME ctype_test1 COMMAND libcu_tests 2)
  add_test(NAME dirent_test1 COMMAND libcu_tests 3)
  add_test(NAME errno_test1 COMMAND libcu_tests 4)
  add_test(NAME fcntl_test1 COMMAND libcu_tests 5)
  add_test(NAME fsystem_test1 COMMAND libcu_tests 6)
  add_test(NAME grp_test1 COMMAND libcu_tests 7)
  add_test(NAME host_test1 COMMAND libcu_tests 8)
  add_test(NAME pwd_test1 COMMAND libcu_tests 9)
  add_test(NAME regex_test1 COMMAND libcu_tests 10)
  add_test(NAME sentinel_test1 COMMAND libcu_tests 11)
  add_test(NAME setjmp_test1 COMMAND libcu_tests 12)
  add_test(NAME stddef_test1 COMMAND libcu_tests 13)
  add_test(NAME stdio_test1 COMMAND libcu_tests 14)
  add_test(NAME stdio_64bit COMMAND libcu_tests 15)
  add_test(NAME stdio_ganging COMMAND libcu_tests 16)
  add_test(NAME stdio_scanf COMMAND libcu_tests 17)
  add_test(NAME stdlib_test1 COMMAND libcu_tests 18)
  add_test(NAME stdlib_strtol COMMAND libcu_tests 19)
  add_test(NAME stdlib_strtoq COMMAND libcu_tests 20)
  add_test(NAME string_test1 COMMAND libcu_tests 21)
  add_test(NAME time_test1 COMMAND libcu_tests 22)
  add_test(NAME unistd_test1 COMMAND libcu_tests 23)

  if (APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath, so that the static cuda runtime can find it at runtime.
    set_property(TARGET libcu_tests PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()
endif()