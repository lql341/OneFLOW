# Optional Hygon DTK (DCU/HIP) build support.
#
# This file only affects builds that ask for the HIP backend. CPU and CI builds
# are untouched: OneFLOW_ConfigureDTK() returns immediately unless a DTK prefix
# is available.
#
# Why it exists
# -------------
# DTK's hip-config.cmake resolves its import prefix from ROCM_PATH:
#
#     set ( _IMPORT_PREFIX "${ROCM_PATH}/hip" "/opt/dtk/hip" )
#
# A configure without ROCM_PATH therefore looks for libgalaxyhip under
# /opt/rocm and stops with "the installation package was faulty ... but not all
# the files it references". In addition, enable_language(HIP) runs
# find_package(amd_comgr) inside a try_compile sub-project, which only inherits
# the variables listed in CMAKE_TRY_COMPILE_PLATFORM_VARIABLES; without that the
# configure stops with "Could not find a package configuration file provided by
# amd_comgr".
#
# Usage
# -----
#     cmake -S . -B build -DONEFLOW_ENABLE_HIP=ON \
#           -DONEFLOW_DTK_ROOT=/path/to/dtk
#
# or export ROCM_PATH=/path/to/dtk and pass -DONEFLOW_ENABLE_HIP=ON (autodetect).
# `scripts/build-dcu.sh` wraps the whole sequence.

set ( ONEFLOW_DTK_ROOT "" CACHE PATH
      "Hygon DTK prefix for the optional HIP backend (empty: do not configure DTK)" )

option ( ONEFLOW_DTK_AUTODETECT
         "Fall back to the ROCM_PATH environment variable when ONEFLOW_DTK_ROOT is empty" ON )

set ( ONEFLOW_DTK_HIP_ARCHITECTURES "" CACHE STRING
      "HIP architectures for the DTK build (empty: keep CMake/device detection)" )

set ( ONEFLOW_DTK_HIP_FLAGS "" CACHE STRING
      "Extra CMAKE_HIP_FLAGS for the DTK build, e.g. --gcc-toolchain=<host gcc> -stdlib=libstdc++ (empty: leave unset)" )

function ( OneFLOW_ConfigureDTK )
    if ( NOT ONEFLOW_ENABLE_HIP AND NOT ONEFLOW_HIP_EXPLICIT )
        return()
    endif ()

    set ( dtk_root "${ONEFLOW_DTK_ROOT}" )
    if ( dtk_root STREQUAL "" AND ONEFLOW_DTK_AUTODETECT )
        set ( dtk_root "$ENV{ROCM_PATH}" )
    endif ()
    if ( dtk_root STREQUAL "" )
        return()
    endif ()
    if ( NOT IS_DIRECTORY "${dtk_root}" )
        message ( FATAL_ERROR
            "ONEFLOW_DTK_ROOT is not a directory: ${dtk_root}" )
    endif ()

    message ( STATUS "OneFLOW DTK prefix     = ${dtk_root}" )

    # hip-config.cmake takes its import prefix from here.
    set ( ENV{ROCM_PATH} "${dtk_root}" )

    # Package search roots shipped by DTK, in the order hip-config.cmake expects.
    set ( dtk_prefixes )
    foreach ( candidate
              "${dtk_root}/dcc/lib64/cmake/AMDDeviceLibs"
              "${dtk_root}/dcc/comgr/lib64/cmake"
              "${dtk_root}"
              "${dtk_root}/.hyhal" )
        if ( IS_DIRECTORY "${candidate}" )
            list ( APPEND dtk_prefixes "${candidate}" )
        endif ()
    endforeach ()
    if ( dtk_prefixes )
        list ( REMOVE_DUPLICATES dtk_prefixes )
        set ( CMAKE_PREFIX_PATH "${dtk_prefixes};${CMAKE_PREFIX_PATH}"
              CACHE STRING "Package search prefixes" FORCE )
        message ( STATUS "OneFLOW DTK prefixes   = ${dtk_prefixes}" )
    endif ()

    # Dependencies hip-config.cmake requires; the try_compile project needs the
    # same values, which is what CMAKE_TRY_COMPILE_PLATFORM_VARIABLES carries.
    set ( dtk_inherited "CMAKE_PREFIX_PATH" )
    foreach ( dep_pair
              "amd_comgr_DIR:${dtk_root}/dcc/comgr/lib64/cmake/amd_comgr"
              "AMDDeviceLibs_DIR:${dtk_root}/dcc/lib64/cmake/AMDDeviceLibs"
              "hsa-runtime64_DIR:${dtk_root}/.hyhal/lib/cmake/hsa-runtime64" )
        string ( REPLACE ":" ";" dep_pair_list "${dep_pair}" )
        list ( GET dep_pair_list 0 dep_name )
        list ( GET dep_pair_list 1 dep_dir )
        if ( IS_DIRECTORY "${dep_dir}" )
            set ( ${dep_name} "${dep_dir}" CACHE PATH
                  "DTK dependency directory discovered by OneFLOWDTK.cmake" FORCE )
            list ( APPEND dtk_inherited "${dep_name}" )
        endif ()
    endforeach ()

    set ( inherited "${CMAKE_TRY_COMPILE_PLATFORM_VARIABLES};${dtk_inherited}" )
    list ( REMOVE_DUPLICATES inherited )
    set ( CMAKE_TRY_COMPILE_PLATFORM_VARIABLES "${inherited}" CACHE STRING
          "Variables forwarded into try_compile projects" FORCE )

    # HIP compiler: DTK ships its own clang front end next to the runtime.
    if ( NOT CMAKE_HIP_COMPILER AND EXISTS "${dtk_root}/llvm/bin/clang++" )
        set ( CMAKE_HIP_COMPILER "${dtk_root}/llvm/bin/clang++" CACHE FILEPATH
              "HIP compiler" FORCE )
        message ( STATUS "OneFLOW HIP compiler   = ${CMAKE_HIP_COMPILER}" )
    endif ()

    if ( NOT ONEFLOW_DTK_HIP_ARCHITECTURES STREQUAL "" )
        set ( CMAKE_HIP_ARCHITECTURES "${ONEFLOW_DTK_HIP_ARCHITECTURES}" CACHE STRING
              "HIP architectures" FORCE )
    endif ()

    if ( NOT ONEFLOW_DTK_HIP_FLAGS STREQUAL "" )
        set ( CMAKE_HIP_FLAGS "${ONEFLOW_DTK_HIP_FLAGS}" CACHE STRING
              "HIP compiler flags" FORCE )
    endif ()
endfunction ()
