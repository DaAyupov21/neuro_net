# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_SOURCE_DIR = /home/damir/CLionProjects/neuron

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/damir/CLionProjects/neuron/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/neuron.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/neuron.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neuron.dir/flags.make

CMakeFiles/neuron.dir/main.cpp.o: CMakeFiles/neuron.dir/flags.make
CMakeFiles/neuron.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/damir/CLionProjects/neuron/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/neuron.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/neuron.dir/main.cpp.o -c /home/damir/CLionProjects/neuron/main.cpp

CMakeFiles/neuron.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuron.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/damir/CLionProjects/neuron/main.cpp > CMakeFiles/neuron.dir/main.cpp.i

CMakeFiles/neuron.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuron.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/damir/CLionProjects/neuron/main.cpp -o CMakeFiles/neuron.dir/main.cpp.s

CMakeFiles/neuron.dir/net.cpp.o: CMakeFiles/neuron.dir/flags.make
CMakeFiles/neuron.dir/net.cpp.o: ../net.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/damir/CLionProjects/neuron/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/neuron.dir/net.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/neuron.dir/net.cpp.o -c /home/damir/CLionProjects/neuron/net.cpp

CMakeFiles/neuron.dir/net.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuron.dir/net.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/damir/CLionProjects/neuron/net.cpp > CMakeFiles/neuron.dir/net.cpp.i

CMakeFiles/neuron.dir/net.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuron.dir/net.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/damir/CLionProjects/neuron/net.cpp -o CMakeFiles/neuron.dir/net.cpp.s

CMakeFiles/neuron.dir/Neuron.cpp.o: CMakeFiles/neuron.dir/flags.make
CMakeFiles/neuron.dir/Neuron.cpp.o: ../Neuron.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/damir/CLionProjects/neuron/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/neuron.dir/Neuron.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/neuron.dir/Neuron.cpp.o -c /home/damir/CLionProjects/neuron/Neuron.cpp

CMakeFiles/neuron.dir/Neuron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuron.dir/Neuron.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/damir/CLionProjects/neuron/Neuron.cpp > CMakeFiles/neuron.dir/Neuron.cpp.i

CMakeFiles/neuron.dir/Neuron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuron.dir/Neuron.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/damir/CLionProjects/neuron/Neuron.cpp -o CMakeFiles/neuron.dir/Neuron.cpp.s

CMakeFiles/neuron.dir/trainingSet.cpp.o: CMakeFiles/neuron.dir/flags.make
CMakeFiles/neuron.dir/trainingSet.cpp.o: ../trainingSet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/damir/CLionProjects/neuron/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/neuron.dir/trainingSet.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/neuron.dir/trainingSet.cpp.o -c /home/damir/CLionProjects/neuron/trainingSet.cpp

CMakeFiles/neuron.dir/trainingSet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuron.dir/trainingSet.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/damir/CLionProjects/neuron/trainingSet.cpp > CMakeFiles/neuron.dir/trainingSet.cpp.i

CMakeFiles/neuron.dir/trainingSet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuron.dir/trainingSet.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/damir/CLionProjects/neuron/trainingSet.cpp -o CMakeFiles/neuron.dir/trainingSet.cpp.s

# Object files for target neuron
neuron_OBJECTS = \
"CMakeFiles/neuron.dir/main.cpp.o" \
"CMakeFiles/neuron.dir/net.cpp.o" \
"CMakeFiles/neuron.dir/Neuron.cpp.o" \
"CMakeFiles/neuron.dir/trainingSet.cpp.o"

# External object files for target neuron
neuron_EXTERNAL_OBJECTS =

neuron: CMakeFiles/neuron.dir/main.cpp.o
neuron: CMakeFiles/neuron.dir/net.cpp.o
neuron: CMakeFiles/neuron.dir/Neuron.cpp.o
neuron: CMakeFiles/neuron.dir/trainingSet.cpp.o
neuron: CMakeFiles/neuron.dir/build.make
neuron: CMakeFiles/neuron.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/damir/CLionProjects/neuron/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable neuron"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neuron.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/neuron.dir/build: neuron

.PHONY : CMakeFiles/neuron.dir/build

CMakeFiles/neuron.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neuron.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neuron.dir/clean

CMakeFiles/neuron.dir/depend:
	cd /home/damir/CLionProjects/neuron/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/damir/CLionProjects/neuron /home/damir/CLionProjects/neuron /home/damir/CLionProjects/neuron/cmake-build-debug /home/damir/CLionProjects/neuron/cmake-build-debug /home/damir/CLionProjects/neuron/cmake-build-debug/CMakeFiles/neuron.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/neuron.dir/depend

