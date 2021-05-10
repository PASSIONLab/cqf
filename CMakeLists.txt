# this was super helpful https://rix0r.nl/blog/2015/08/13/cmake-guide/
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(cqf_gpu VERSION 0.1 LANGUAGES C CUDA)

# Must use GNUInstallDirs to install libraries into correct
# locations on all platforms.
include(GNUInstallDirs)

# Include Boost as an imported target
#find_package(Boost REQUIRED)
#add_library(boost INTERFACE IMPORTED)
#set_property(TARGET boost PROPERTY
#    INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})

# Some other library that we import that was also built using CMake
# and has an exported target.
#find_package(MyOtherLibrary REQUIRED)

# Targets that we develop here
enable_testing()
add_subdirectory(liblib)
#add_subdirectory(app)
