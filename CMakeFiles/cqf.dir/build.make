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
include CMakeFiles/cqf.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cqf.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cqf.dir/flags.make

CMakeFiles/cqf.dir/src/bm.c.o: CMakeFiles/cqf.dir/flags.make
CMakeFiles/cqf.dir/src/bm.c.o: src/bm.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/cqf.dir/src/bm.c.o"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cqf.dir/src/bm.c.o   -c /global/homes/e/ewims/cqf/src/bm.c

CMakeFiles/cqf.dir/src/bm.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cqf.dir/src/bm.c.i"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/homes/e/ewims/cqf/src/bm.c > CMakeFiles/cqf.dir/src/bm.c.i

CMakeFiles/cqf.dir/src/bm.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cqf.dir/src/bm.c.s"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/homes/e/ewims/cqf/src/bm.c -o CMakeFiles/cqf.dir/src/bm.c.s

CMakeFiles/cqf.dir/src/gqf_file.c.o: CMakeFiles/cqf.dir/flags.make
CMakeFiles/cqf.dir/src/gqf_file.c.o: src/gqf_file.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/cqf.dir/src/gqf_file.c.o"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cqf.dir/src/gqf_file.c.o   -c /global/homes/e/ewims/cqf/src/gqf_file.c

CMakeFiles/cqf.dir/src/gqf_file.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cqf.dir/src/gqf_file.c.i"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/homes/e/ewims/cqf/src/gqf_file.c > CMakeFiles/cqf.dir/src/gqf_file.c.i

CMakeFiles/cqf.dir/src/gqf_file.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cqf.dir/src/gqf_file.c.s"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/homes/e/ewims/cqf/src/gqf_file.c -o CMakeFiles/cqf.dir/src/gqf_file.c.s

CMakeFiles/cqf.dir/src/hashutil.c.o: CMakeFiles/cqf.dir/flags.make
CMakeFiles/cqf.dir/src/hashutil.c.o: src/hashutil.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/cqf.dir/src/hashutil.c.o"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cqf.dir/src/hashutil.c.o   -c /global/homes/e/ewims/cqf/src/hashutil.c

CMakeFiles/cqf.dir/src/hashutil.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cqf.dir/src/hashutil.c.i"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/homes/e/ewims/cqf/src/hashutil.c > CMakeFiles/cqf.dir/src/hashutil.c.i

CMakeFiles/cqf.dir/src/hashutil.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cqf.dir/src/hashutil.c.s"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/homes/e/ewims/cqf/src/hashutil.c -o CMakeFiles/cqf.dir/src/hashutil.c.s

CMakeFiles/cqf.dir/src/partitioned_counter.c.o: CMakeFiles/cqf.dir/flags.make
CMakeFiles/cqf.dir/src/partitioned_counter.c.o: src/partitioned_counter.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/cqf.dir/src/partitioned_counter.c.o"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cqf.dir/src/partitioned_counter.c.o   -c /global/homes/e/ewims/cqf/src/partitioned_counter.c

CMakeFiles/cqf.dir/src/partitioned_counter.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cqf.dir/src/partitioned_counter.c.i"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/homes/e/ewims/cqf/src/partitioned_counter.c > CMakeFiles/cqf.dir/src/partitioned_counter.c.i

CMakeFiles/cqf.dir/src/partitioned_counter.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cqf.dir/src/partitioned_counter.c.s"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/homes/e/ewims/cqf/src/partitioned_counter.c -o CMakeFiles/cqf.dir/src/partitioned_counter.c.s

CMakeFiles/cqf.dir/src/gqf.cu.o: CMakeFiles/cqf.dir/flags.make
CMakeFiles/cqf.dir/src/gqf.cu.o: src/gqf.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/cqf.dir/src/gqf.cu.o"
	/usr/common/software/sles15_cgpu/cuda/11.1.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /global/homes/e/ewims/cqf/src/gqf.cu -o CMakeFiles/cqf.dir/src/gqf.cu.o

CMakeFiles/cqf.dir/src/gqf.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cqf.dir/src/gqf.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cqf.dir/src/gqf.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cqf.dir/src/gqf.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cqf.dir/src/test_partitioned_counter.c.o: CMakeFiles/cqf.dir/flags.make
CMakeFiles/cqf.dir/src/test_partitioned_counter.c.o: src/test_partitioned_counter.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/cqf.dir/src/test_partitioned_counter.c.o"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cqf.dir/src/test_partitioned_counter.c.o   -c /global/homes/e/ewims/cqf/src/test_partitioned_counter.c

CMakeFiles/cqf.dir/src/test_partitioned_counter.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cqf.dir/src/test_partitioned_counter.c.i"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/homes/e/ewims/cqf/src/test_partitioned_counter.c > CMakeFiles/cqf.dir/src/test_partitioned_counter.c.i

CMakeFiles/cqf.dir/src/test_partitioned_counter.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cqf.dir/src/test_partitioned_counter.c.s"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/homes/e/ewims/cqf/src/test_partitioned_counter.c -o CMakeFiles/cqf.dir/src/test_partitioned_counter.c.s

CMakeFiles/cqf.dir/src/test_threadsafe.c.o: CMakeFiles/cqf.dir/flags.make
CMakeFiles/cqf.dir/src/test_threadsafe.c.o: src/test_threadsafe.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/cqf.dir/src/test_threadsafe.c.o"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cqf.dir/src/test_threadsafe.c.o   -c /global/homes/e/ewims/cqf/src/test_threadsafe.c

CMakeFiles/cqf.dir/src/test_threadsafe.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cqf.dir/src/test_threadsafe.c.i"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/homes/e/ewims/cqf/src/test_threadsafe.c > CMakeFiles/cqf.dir/src/test_threadsafe.c.i

CMakeFiles/cqf.dir/src/test_threadsafe.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cqf.dir/src/test_threadsafe.c.s"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/homes/e/ewims/cqf/src/test_threadsafe.c -o CMakeFiles/cqf.dir/src/test_threadsafe.c.s

CMakeFiles/cqf.dir/src/zipf.c.o: CMakeFiles/cqf.dir/flags.make
CMakeFiles/cqf.dir/src/zipf.c.o: src/zipf.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object CMakeFiles/cqf.dir/src/zipf.c.o"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cqf.dir/src/zipf.c.o   -c /global/homes/e/ewims/cqf/src/zipf.c

CMakeFiles/cqf.dir/src/zipf.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cqf.dir/src/zipf.c.i"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /global/homes/e/ewims/cqf/src/zipf.c > CMakeFiles/cqf.dir/src/zipf.c.i

CMakeFiles/cqf.dir/src/zipf.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cqf.dir/src/zipf.c.s"
	/opt/cray/pe/craype/2.6.2/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /global/homes/e/ewims/cqf/src/zipf.c -o CMakeFiles/cqf.dir/src/zipf.c.s

# Object files for target cqf
cqf_OBJECTS = \
"CMakeFiles/cqf.dir/src/bm.c.o" \
"CMakeFiles/cqf.dir/src/gqf_file.c.o" \
"CMakeFiles/cqf.dir/src/hashutil.c.o" \
"CMakeFiles/cqf.dir/src/partitioned_counter.c.o" \
"CMakeFiles/cqf.dir/src/gqf.cu.o" \
"CMakeFiles/cqf.dir/src/test_partitioned_counter.c.o" \
"CMakeFiles/cqf.dir/src/test_threadsafe.c.o" \
"CMakeFiles/cqf.dir/src/zipf.c.o"

# External object files for target cqf
cqf_EXTERNAL_OBJECTS =

libcqf.a: CMakeFiles/cqf.dir/src/bm.c.o
libcqf.a: CMakeFiles/cqf.dir/src/gqf_file.c.o
libcqf.a: CMakeFiles/cqf.dir/src/hashutil.c.o
libcqf.a: CMakeFiles/cqf.dir/src/partitioned_counter.c.o
libcqf.a: CMakeFiles/cqf.dir/src/gqf.cu.o
libcqf.a: CMakeFiles/cqf.dir/src/test_partitioned_counter.c.o
libcqf.a: CMakeFiles/cqf.dir/src/test_threadsafe.c.o
libcqf.a: CMakeFiles/cqf.dir/src/zipf.c.o
libcqf.a: CMakeFiles/cqf.dir/build.make
libcqf.a: CMakeFiles/cqf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/homes/e/ewims/cqf/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CUDA static library libcqf.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cqf.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cqf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cqf.dir/build: libcqf.a

.PHONY : CMakeFiles/cqf.dir/build

CMakeFiles/cqf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cqf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cqf.dir/clean

CMakeFiles/cqf.dir/depend:
	cd /global/homes/e/ewims/cqf && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/homes/e/ewims/cqf /global/homes/e/ewims/cqf /global/homes/e/ewims/cqf /global/homes/e/ewims/cqf /global/homes/e/ewims/cqf/CMakeFiles/cqf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cqf.dir/depend
