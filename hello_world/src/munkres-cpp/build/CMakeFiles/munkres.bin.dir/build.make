# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_COMMAND = /ihome/crc/install/cmake/3.7.1/bin/cmake

# The command to remove a file.
RM = /ihome/crc/install/cmake/3.7.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/munkres.bin.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/munkres.bin.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/munkres.bin.dir/flags.make

CMakeFiles/munkres.bin.dir/examples/main.cpp.o: CMakeFiles/munkres.bin.dir/flags.make
CMakeFiles/munkres.bin.dir/examples/main.cpp.o: ../examples/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/ihome/ageorge/plh25/SummerMHTv1/munkres-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/munkres.bin.dir/examples/main.cpp.o"
	/ihome/crc/install/gcc/8.2.0/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/munkres.bin.dir/examples/main.cpp.o -c /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp/examples/main.cpp

CMakeFiles/munkres.bin.dir/examples/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/munkres.bin.dir/examples/main.cpp.i"
	/ihome/crc/install/gcc/8.2.0/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp/examples/main.cpp > CMakeFiles/munkres.bin.dir/examples/main.cpp.i

CMakeFiles/munkres.bin.dir/examples/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/munkres.bin.dir/examples/main.cpp.s"
	/ihome/crc/install/gcc/8.2.0/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp/examples/main.cpp -o CMakeFiles/munkres.bin.dir/examples/main.cpp.s

CMakeFiles/munkres.bin.dir/examples/main.cpp.o.requires:

.PHONY : CMakeFiles/munkres.bin.dir/examples/main.cpp.o.requires

CMakeFiles/munkres.bin.dir/examples/main.cpp.o.provides: CMakeFiles/munkres.bin.dir/examples/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/munkres.bin.dir/build.make CMakeFiles/munkres.bin.dir/examples/main.cpp.o.provides.build
.PHONY : CMakeFiles/munkres.bin.dir/examples/main.cpp.o.provides

CMakeFiles/munkres.bin.dir/examples/main.cpp.o.provides.build: CMakeFiles/munkres.bin.dir/examples/main.cpp.o


# Object files for target munkres.bin
munkres_bin_OBJECTS = \
"CMakeFiles/munkres.bin.dir/examples/main.cpp.o"

# External object files for target munkres.bin
munkres_bin_EXTERNAL_OBJECTS =

munkres.bin: CMakeFiles/munkres.bin.dir/examples/main.cpp.o
munkres.bin: CMakeFiles/munkres.bin.dir/build.make
munkres.bin: libmunkres.a
munkres.bin: CMakeFiles/munkres.bin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/ihome/ageorge/plh25/SummerMHTv1/munkres-cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable munkres.bin"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/munkres.bin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/munkres.bin.dir/build: munkres.bin

.PHONY : CMakeFiles/munkres.bin.dir/build

CMakeFiles/munkres.bin.dir/requires: CMakeFiles/munkres.bin.dir/examples/main.cpp.o.requires

.PHONY : CMakeFiles/munkres.bin.dir/requires

CMakeFiles/munkres.bin.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/munkres.bin.dir/cmake_clean.cmake
.PHONY : CMakeFiles/munkres.bin.dir/clean

CMakeFiles/munkres.bin.dir/depend:
	cd /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp/build /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp/build /ihome/ageorge/plh25/SummerMHTv1/munkres-cpp/build/CMakeFiles/munkres.bin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/munkres.bin.dir/depend

