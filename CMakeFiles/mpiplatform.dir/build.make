# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/MPIPlatform4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/MPIPlatform4

# Include any dependencies generated for this target.
include CMakeFiles/mpiplatform.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mpiplatform.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpiplatform.dir/flags.make

CMakeFiles/mpiplatform.dir/src/main.cpp.o: CMakeFiles/mpiplatform.dir/flags.make
CMakeFiles/mpiplatform.dir/src/main.cpp.o: src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/MPIPlatform4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mpiplatform.dir/src/main.cpp.o"
	/usr/bin/mpiCC   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mpiplatform.dir/src/main.cpp.o -c /home/ubuntu/MPIPlatform4/src/main.cpp

CMakeFiles/mpiplatform.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mpiplatform.dir/src/main.cpp.i"
	/usr/bin/mpiCC  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/MPIPlatform4/src/main.cpp > CMakeFiles/mpiplatform.dir/src/main.cpp.i

CMakeFiles/mpiplatform.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mpiplatform.dir/src/main.cpp.s"
	/usr/bin/mpiCC  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/MPIPlatform4/src/main.cpp -o CMakeFiles/mpiplatform.dir/src/main.cpp.s

CMakeFiles/mpiplatform.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/mpiplatform.dir/src/main.cpp.o.requires

CMakeFiles/mpiplatform.dir/src/main.cpp.o.provides: CMakeFiles/mpiplatform.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/mpiplatform.dir/build.make CMakeFiles/mpiplatform.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/mpiplatform.dir/src/main.cpp.o.provides

CMakeFiles/mpiplatform.dir/src/main.cpp.o.provides.build: CMakeFiles/mpiplatform.dir/src/main.cpp.o


# Object files for target mpiplatform
mpiplatform_OBJECTS = \
"CMakeFiles/mpiplatform.dir/src/main.cpp.o"

# External object files for target mpiplatform
mpiplatform_EXTERNAL_OBJECTS =

mpiplatform: CMakeFiles/mpiplatform.dir/src/main.cpp.o
mpiplatform: CMakeFiles/mpiplatform.dir/build.make
mpiplatform: gflags/libgflags_nothreads.a
mpiplatform: CMakeFiles/mpiplatform.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/MPIPlatform4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mpiplatform"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpiplatform.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpiplatform.dir/build: mpiplatform

.PHONY : CMakeFiles/mpiplatform.dir/build

CMakeFiles/mpiplatform.dir/requires: CMakeFiles/mpiplatform.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/mpiplatform.dir/requires

CMakeFiles/mpiplatform.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpiplatform.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpiplatform.dir/clean

CMakeFiles/mpiplatform.dir/depend:
	cd /home/ubuntu/MPIPlatform4 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/MPIPlatform4 /home/ubuntu/MPIPlatform4 /home/ubuntu/MPIPlatform4 /home/ubuntu/MPIPlatform4 /home/ubuntu/MPIPlatform4/CMakeFiles/mpiplatform.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpiplatform.dir/depend

