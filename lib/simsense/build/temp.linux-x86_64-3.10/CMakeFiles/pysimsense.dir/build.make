# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bart/git/Shapebench/lib/simsense

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bart/git/Shapebench/lib/simsense/build/temp.linux-x86_64-3.10

# Include any dependencies generated for this target.
include CMakeFiles/pysimsense.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pysimsense.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pysimsense.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pysimsense.dir/flags.make

CMakeFiles/pysimsense.dir/python/pysimsense.cpp.o: CMakeFiles/pysimsense.dir/flags.make
CMakeFiles/pysimsense.dir/python/pysimsense.cpp.o: ../../python/pysimsense.cpp
CMakeFiles/pysimsense.dir/python/pysimsense.cpp.o: CMakeFiles/pysimsense.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bart/git/Shapebench/lib/simsense/build/temp.linux-x86_64-3.10/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pysimsense.dir/python/pysimsense.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pysimsense.dir/python/pysimsense.cpp.o -MF CMakeFiles/pysimsense.dir/python/pysimsense.cpp.o.d -o CMakeFiles/pysimsense.dir/python/pysimsense.cpp.o -c /home/bart/git/Shapebench/lib/simsense/python/pysimsense.cpp

CMakeFiles/pysimsense.dir/python/pysimsense.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pysimsense.dir/python/pysimsense.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bart/git/Shapebench/lib/simsense/python/pysimsense.cpp > CMakeFiles/pysimsense.dir/python/pysimsense.cpp.i

CMakeFiles/pysimsense.dir/python/pysimsense.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pysimsense.dir/python/pysimsense.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bart/git/Shapebench/lib/simsense/python/pysimsense.cpp -o CMakeFiles/pysimsense.dir/python/pysimsense.cpp.s

# Object files for target pysimsense
pysimsense_OBJECTS = \
"CMakeFiles/pysimsense.dir/python/pysimsense.cpp.o"

# External object files for target pysimsense
pysimsense_EXTERNAL_OBJECTS =

../lib.linux-x86_64-3.10/simsense/pysimsense.cpython-310-x86_64-linux-gnu.so: CMakeFiles/pysimsense.dir/python/pysimsense.cpp.o
../lib.linux-x86_64-3.10/simsense/pysimsense.cpython-310-x86_64-linux-gnu.so: CMakeFiles/pysimsense.dir/build.make
../lib.linux-x86_64-3.10/simsense/pysimsense.cpython-310-x86_64-linux-gnu.so: ../lib.linux-x86_64-3.10/simsense/libsimsense.so
../lib.linux-x86_64-3.10/simsense/pysimsense.cpython-310-x86_64-linux-gnu.so: CMakeFiles/pysimsense.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bart/git/Shapebench/lib/simsense/build/temp.linux-x86_64-3.10/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ../lib.linux-x86_64-3.10/simsense/pysimsense.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pysimsense.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/bart/git/Shapebench/lib/simsense/build/lib.linux-x86_64-3.10/simsense/pysimsense.cpython-310-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/pysimsense.dir/build: ../lib.linux-x86_64-3.10/simsense/pysimsense.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/pysimsense.dir/build

CMakeFiles/pysimsense.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pysimsense.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pysimsense.dir/clean

CMakeFiles/pysimsense.dir/depend:
	cd /home/bart/git/Shapebench/lib/simsense/build/temp.linux-x86_64-3.10 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bart/git/Shapebench/lib/simsense /home/bart/git/Shapebench/lib/simsense /home/bart/git/Shapebench/lib/simsense/build/temp.linux-x86_64-3.10 /home/bart/git/Shapebench/lib/simsense/build/temp.linux-x86_64-3.10 /home/bart/git/Shapebench/lib/simsense/build/temp.linux-x86_64-3.10/CMakeFiles/pysimsense.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pysimsense.dir/depend
