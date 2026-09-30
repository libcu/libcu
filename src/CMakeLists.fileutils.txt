cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(libcu LANGUAGES CXX CUDA)

include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
link_directories(../_build/build64)

# dcat
add_executable(dcat
	libcu.fileutils/dcat.cpp
  )
# dcd
add_executable(dcd
	libcu.fileutils/dcd.cpp
  )
# dchgrp
add_executable(dchgrp
	libcu.fileutils/dchgrp.cpp
  )
# dchmod
add_executable(dchmod
	libcu.fileutils/dchmod.cpp
  )
# dchown
add_executable(dchown
	libcu.fileutils/dchown.cpp
  )
# dcmp
add_executable(dcmp
	libcu.fileutils/dcmp.cpp
  )
# dcp
add_executable(dcp
	libcu.fileutils/dcp.cpp
  )
# dgrep
add_executable(dgrep
	libcu.fileutils/dgrep.cpp
  )
# dls
add_executable(dls
	libcu.fileutils/dls.cpp
  )
# dmkdir
add_executable(dmkdir
	libcu.fileutils/dmkdir.cpp
  )
# dmore
add_executable(dmore
	libcu.fileutils/dmore.cpp
  )
# dmv
add_executable(dmv
	libcu.fileutils/dmv.cpp
  )
# dpwd
add_executable(dpwd
	libcu.fileutils/dpwd.cpp
  )
# drm
add_executable(drm
	libcu.fileutils/drm.cpp
  )
# drmdir
add_executable(drmdir
	libcu.fileutils/drmdir.cpp
  )

# Request that libcu be built with -std=c++11. As this is a public compile feature anything that links to particles will also build with -std=c++11
target_compile_features(dcat PUBLIC cxx_std_11)
target_compile_features(dcd PUBLIC cxx_std_11)
target_compile_features(dchgrp PUBLIC cxx_std_11)
target_compile_features(dchmod PUBLIC cxx_std_11)
target_compile_features(dchown PUBLIC cxx_std_11)
target_compile_features(dcmp PUBLIC cxx_std_11)
target_compile_features(dcp PUBLIC cxx_std_11)
target_compile_features(dgrep PUBLIC cxx_std_11)
target_compile_features(dls PUBLIC cxx_std_11)
target_compile_features(dmkdir PUBLIC cxx_std_11)
target_compile_features(dmore PUBLIC cxx_std_11)
target_compile_features(dmv PUBLIC cxx_std_11)
target_compile_features(dpwd PUBLIC cxx_std_11)
target_compile_features(drm PUBLIC cxx_std_11)
target_compile_features(drmdir PUBLIC cxx_std_11)

if (M)
  set_target_properties(dcat PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dcd PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dchgrp PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dchmod PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dchown PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dcp PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dgrep PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dls PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dmkdir PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dmore PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dmv PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(dpwd PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(drm PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
  set_target_properties(drmdir PROPERTIES COMPILE_FLAGS "-m${M}" LINK_FLAGS "-m${M}")
endif()