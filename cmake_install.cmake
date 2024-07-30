# Install script for directory: D:/FastNoise2

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/FastNoise2")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/Microsoft/AndroidNDK/android-ndk-r23c/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libFastNoise.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libFastNoise.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libFastNoise.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "D:/FastNoise2/src/libFastNoise.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libFastNoise.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libFastNoise.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "C:/Microsoft/AndroidNDK/android-ndk-r23c/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-strip.exe" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libFastNoise.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/FastSIMD" TYPE FILE FILES
    "D:/FastNoise2/src/../include/FastSIMD/FastSIMD.h"
    "D:/FastNoise2/src/../include/FastSIMD/FastSIMD_Config.h"
    "D:/FastNoise2/src/../include/FastSIMD/FastSIMD_Export.h"
    "D:/FastNoise2/src/../include/FastSIMD/FunctionList.h"
    "D:/FastNoise2/src/../include/FastSIMD/InlInclude.h"
    "D:/FastNoise2/src/../include/FastSIMD/SIMDTypeList.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/FastNoise" TYPE FILE FILES
    "D:/FastNoise2/src/../include/FastNoise/FastNoise.h"
    "D:/FastNoise2/src/../include/FastNoise/FastNoise_C.h"
    "D:/FastNoise2/src/../include/FastNoise/FastNoise_Config.h"
    "D:/FastNoise2/src/../include/FastNoise/FastNoise_Export.h"
    "D:/FastNoise2/src/../include/FastNoise/Metadata.h"
    "D:/FastNoise2/src/../include/FastNoise/SmartNode.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/FastNoise/Generators" TYPE FILE FILES
    "D:/FastNoise2/src/../include/FastNoise/Generators/BasicGenerators.h"
    "D:/FastNoise2/src/../include/FastNoise/Generators/Blends.h"
    "D:/FastNoise2/src/../include/FastNoise/Generators/Cellular.h"
    "D:/FastNoise2/src/../include/FastNoise/Generators/DomainWarp.h"
    "D:/FastNoise2/src/../include/FastNoise/Generators/DomainWarpFractal.h"
    "D:/FastNoise2/src/../include/FastNoise/Generators/Fractal.h"
    "D:/FastNoise2/src/../include/FastNoise/Generators/Generator.h"
    "D:/FastNoise2/src/../include/FastNoise/Generators/Modifiers.h"
    "D:/FastNoise2/src/../include/FastNoise/Generators/Perlin.h"
    "D:/FastNoise2/src/../include/FastNoise/Generators/Simplex.h"
    "D:/FastNoise2/src/../include/FastNoise/Generators/Value.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/FastNoise2" TYPE FILE FILES
    "D:/FastNoise2/generated/FastNoise2Config.cmake"
    "D:/FastNoise2/generated/FastNoise2ConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/FastNoise2/FastNoise2Targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/FastNoise2/FastNoise2Targets.cmake"
         "D:/FastNoise2/CMakeFiles/Export/685b7b2fe36c8f3a2e6ec6f0a6279b14/FastNoise2Targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/FastNoise2/FastNoise2Targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/FastNoise2/FastNoise2Targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/FastNoise2" TYPE FILE FILES "D:/FastNoise2/CMakeFiles/Export/685b7b2fe36c8f3a2e6ec6f0a6279b14/FastNoise2Targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/FastNoise2" TYPE FILE FILES "D:/FastNoise2/CMakeFiles/Export/685b7b2fe36c8f3a2e6ec6f0a6279b14/FastNoise2Targets-noconfig.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("D:/FastNoise2/src/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
  file(WRITE "D:/FastNoise2/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
