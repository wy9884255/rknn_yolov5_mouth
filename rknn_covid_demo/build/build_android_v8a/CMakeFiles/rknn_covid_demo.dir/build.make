# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

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
CMAKE_COMMAND = /home/abc/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/abc/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/abc/chip2/rknpu2/examples/rknn_covid_demo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/abc/chip2/rknpu2/examples/rknn_covid_demo/build/build_android_v8a

# Include any dependencies generated for this target.
include CMakeFiles/rknn_covid_demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rknn_covid_demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rknn_covid_demo.dir/flags.make

CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.o: CMakeFiles/rknn_covid_demo.dir/flags.make
CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.o: ../../src/postprocess.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/abc/chip2/rknpu2/examples/rknn_covid_demo/build/build_android_v8a/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.o"
	/home/abc/chip/android-ndk-r17c/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-g++ --sysroot=/home/abc/chip/android-ndk-r17c/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.o -c /home/abc/chip2/rknpu2/examples/rknn_covid_demo/src/postprocess.cc

CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.i"
	/home/abc/chip/android-ndk-r17c/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-g++ --sysroot=/home/abc/chip/android-ndk-r17c/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/abc/chip2/rknpu2/examples/rknn_covid_demo/src/postprocess.cc > CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.i

CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.s"
	/home/abc/chip/android-ndk-r17c/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-g++ --sysroot=/home/abc/chip/android-ndk-r17c/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/abc/chip2/rknpu2/examples/rknn_covid_demo/src/postprocess.cc -o CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.s

CMakeFiles/rknn_covid_demo.dir/src/api.cc.o: CMakeFiles/rknn_covid_demo.dir/flags.make
CMakeFiles/rknn_covid_demo.dir/src/api.cc.o: ../../src/api.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/abc/chip2/rknpu2/examples/rknn_covid_demo/build/build_android_v8a/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/rknn_covid_demo.dir/src/api.cc.o"
	/home/abc/chip/android-ndk-r17c/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-g++ --sysroot=/home/abc/chip/android-ndk-r17c/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rknn_covid_demo.dir/src/api.cc.o -c /home/abc/chip2/rknpu2/examples/rknn_covid_demo/src/api.cc

CMakeFiles/rknn_covid_demo.dir/src/api.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rknn_covid_demo.dir/src/api.cc.i"
	/home/abc/chip/android-ndk-r17c/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-g++ --sysroot=/home/abc/chip/android-ndk-r17c/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/abc/chip2/rknpu2/examples/rknn_covid_demo/src/api.cc > CMakeFiles/rknn_covid_demo.dir/src/api.cc.i

CMakeFiles/rknn_covid_demo.dir/src/api.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rknn_covid_demo.dir/src/api.cc.s"
	/home/abc/chip/android-ndk-r17c/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-g++ --sysroot=/home/abc/chip/android-ndk-r17c/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/abc/chip2/rknpu2/examples/rknn_covid_demo/src/api.cc -o CMakeFiles/rknn_covid_demo.dir/src/api.cc.s

# Object files for target rknn_covid_demo
rknn_covid_demo_OBJECTS = \
"CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.o" \
"CMakeFiles/rknn_covid_demo.dir/src/api.cc.o"

# External object files for target rknn_covid_demo
rknn_covid_demo_EXTERNAL_OBJECTS =

librknn_covid_demo.so: CMakeFiles/rknn_covid_demo.dir/src/postprocess.cc.o
librknn_covid_demo.so: CMakeFiles/rknn_covid_demo.dir/src/api.cc.o
librknn_covid_demo.so: CMakeFiles/rknn_covid_demo.dir/build.make
librknn_covid_demo.so: ../../../../runtime/RK356X/Android/librknn_api/arm64-v8a/librknnrt.so
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_calib3d.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_core.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_features2d.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_imgcodecs.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_imgproc.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/staticlibs/arm64-v8a/libopencv_core.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/libtegra_hal.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/liblibjpeg-turbo.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/liblibwebp.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/libcpufeatures.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/liblibpng.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/liblibtiff.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/liblibjasper.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/libIlmImf.a
librknn_covid_demo.so: /home/abc/chip2/rknpu2/examples/3rdparty/opencv/OpenCV-android-sdk/sdk/native/3rdparty/libs/arm64-v8a/libzlib.a
librknn_covid_demo.so: CMakeFiles/rknn_covid_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/abc/chip2/rknpu2/examples/rknn_covid_demo/build/build_android_v8a/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library librknn_covid_demo.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rknn_covid_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rknn_covid_demo.dir/build: librknn_covid_demo.so

.PHONY : CMakeFiles/rknn_covid_demo.dir/build

CMakeFiles/rknn_covid_demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rknn_covid_demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rknn_covid_demo.dir/clean

CMakeFiles/rknn_covid_demo.dir/depend:
	cd /home/abc/chip2/rknpu2/examples/rknn_covid_demo/build/build_android_v8a && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/abc/chip2/rknpu2/examples/rknn_covid_demo /home/abc/chip2/rknpu2/examples/rknn_covid_demo /home/abc/chip2/rknpu2/examples/rknn_covid_demo/build/build_android_v8a /home/abc/chip2/rknpu2/examples/rknn_covid_demo/build/build_android_v8a /home/abc/chip2/rknpu2/examples/rknn_covid_demo/build/build_android_v8a/CMakeFiles/rknn_covid_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rknn_covid_demo.dir/depend

