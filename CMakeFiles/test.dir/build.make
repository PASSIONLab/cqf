# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /global/common/sw/cray/cnl7/haswell/cmake/3.14.4/gcc/8.2.0/2hef55n/bin/cmake

# The command to remove a file.
RM = /global/common/sw/cray/cnl7/haswell/cmake/3.14.4/gcc/8.2.0/2hef55n/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /global/homes/e/ewims/cqf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /global/homes/e/ewims/cqf

# Include any dependencies generated for this target.
include CMakeFiles/test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test.dir/flags.make

CMakeFiles/test.dir/src/test.cu.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/src/test.cu.o: src/test.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/test.dir/src/test.cu.o"
	/usr/common/software/sles15_cgpu/cuda/11.1.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /global/homes/e/ewims/cqf/src/test.cu -o CMakeFiles/test.dir/src/test.cu.o

CMakeFiles/test.dir/src/test.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/test.dir/src/test.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/test.dir/src/test.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/test.dir/src/test.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/src/test.cu.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

CMakeFiles/test.dir/cmake_device_link.o: CMakeFiles/test.dir/src/test.cu.o
CMakeFiles/test.dir/cmake_device_link.o: CMakeFiles/test.dir/build.make
CMakeFiles/test.dir/cmake_device_link.o: CMakeFiles/test.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/test.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: CMakeFiles/test.dir/cmake_device_link.o

.PHONY : CMakeFiles/test.dir/build

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/src/test.cu.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

test: CMakeFiles/test.dir/src/test.cu.o
test: CMakeFiles/test.dir/build.make
test: CMakeFiles/test.dir/cmake_device_link.o
test: CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: test

.PHONY : CMakeFiles/test.dir/build

CMakeFiles/test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test.dir/clean

CMakeFiles/test.dir/depend:
	cd /global/homes/e/ewims/cqf && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/homes/e/ewims/cqf /global/homes/e/ewims/cqf /global/homes/e/ewims/cqf /global/homes/e/ewims/cqf /global/homes/e/ewims/cqf/CMakeFiles/test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test.dir/depend
