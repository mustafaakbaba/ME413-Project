# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/betul/ME413/dealii/project/proj/part2o2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/betul/ME413/dealii/project/proj/part2o2

# Include any dependencies generated for this target.
include CMakeFiles/part2o2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/part2o2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/part2o2.dir/flags.make

CMakeFiles/part2o2.dir/part2o2.cc.o: CMakeFiles/part2o2.dir/flags.make
CMakeFiles/part2o2.dir/part2o2.cc.o: part2o2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/betul/ME413/dealii/project/proj/part2o2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/part2o2.dir/part2o2.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/part2o2.dir/part2o2.cc.o -c /home/betul/ME413/dealii/project/proj/part2o2/part2o2.cc

CMakeFiles/part2o2.dir/part2o2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/part2o2.dir/part2o2.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/betul/ME413/dealii/project/proj/part2o2/part2o2.cc > CMakeFiles/part2o2.dir/part2o2.cc.i

CMakeFiles/part2o2.dir/part2o2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/part2o2.dir/part2o2.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/betul/ME413/dealii/project/proj/part2o2/part2o2.cc -o CMakeFiles/part2o2.dir/part2o2.cc.s

# Object files for target part2o2
part2o2_OBJECTS = \
"CMakeFiles/part2o2.dir/part2o2.cc.o"

# External object files for target part2o2
part2o2_EXTERNAL_OBJECTS =

part2o2: CMakeFiles/part2o2.dir/part2o2.cc.o
part2o2: CMakeFiles/part2o2.dir/build.make
part2o2: /home/betul/ME413/dealii/lib/libdeal_II.g.so.9.5.0-pre
part2o2: /usr/lib/x86_64-linux-gnu/libtbb.so
part2o2: /usr/lib/x86_64-linux-gnu/libz.so
part2o2: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
part2o2: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
part2o2: /usr/lib/x86_64-linux-gnu/libboost_system.so
part2o2: /usr/lib/x86_64-linux-gnu/libboost_thread.so
part2o2: /usr/lib/x86_64-linux-gnu/libboost_regex.so
part2o2: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
part2o2: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
part2o2: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
part2o2: /usr/lib/x86_64-linux-gnu/libarpack.so
part2o2: /usr/lib/x86_64-linux-gnu/liblapack.so
part2o2: /usr/lib/x86_64-linux-gnu/libblas.so
part2o2: /usr/lib/x86_64-linux-gnu/libassimp.so
part2o2: CMakeFiles/part2o2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/betul/ME413/dealii/project/proj/part2o2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable part2o2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/part2o2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/part2o2.dir/build: part2o2

.PHONY : CMakeFiles/part2o2.dir/build

CMakeFiles/part2o2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/part2o2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/part2o2.dir/clean

CMakeFiles/part2o2.dir/depend:
	cd /home/betul/ME413/dealii/project/proj/part2o2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/betul/ME413/dealii/project/proj/part2o2 /home/betul/ME413/dealii/project/proj/part2o2 /home/betul/ME413/dealii/project/proj/part2o2 /home/betul/ME413/dealii/project/proj/part2o2 /home/betul/ME413/dealii/project/proj/part2o2/CMakeFiles/part2o2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/part2o2.dir/depend

