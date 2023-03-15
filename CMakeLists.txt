cmake_minimum_required(VERSION 3.12)
project(Life VERSION 0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ======== GLAD ========
add_subdirectory(dependencies/glad/cmake)
glad_add_library(glad STATIC API gl:core=4.6)

# ======== GLFW ========
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(dependencies/glfw)

# ======== GLM ========
add_library(
	glm INTERFACE
)
target_include_directories(glm INTERFACE dependencies/glm)

# ======== Life-GUI ========
add_executable(
	life-gui
	solver.hpp
	reference.hpp
	main-gui.cpp
	utils.cpp
)
target_link_libraries(life-gui PUBLIC glad glfw glm)
target_include_directories(life-gui PUBLIC . dependencies/)

# ======== Life-Benchmark ========
add_executable(
	life-benchmark
	solver.hpp
	reference.hpp
	main-benchmark.cpp
	utils.cpp
)
target_include_directories(life-benchmark PUBLIC .)