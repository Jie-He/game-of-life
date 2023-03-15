# Game of Life

Starter kit for the practical component of the lesson on multithreading.

## Prerequisites:
1. Git
1. CMake
1. Python (+ `jinja2`)

## Setup:
1. Clone repository with submodules.
	1. `git clone --recurse-submodules`
	1. Or after cloning `git submodule update --init`
1. Make sure `jinja2` is installed: `python -m pip install jinja2`
1. `mkdir build && cd build`
1. `cmake ..`
1. Open `build\Life.sln` in Visual Studio

## Usage
1. Implement the calculation of the Game of Life in `solver.hpp`.
	1. There are different API versions available. See below for details.
1. Run life-gui to see your implementation in action
	1. Configure the grid size by modifying `constexpr int dim = 258;` in main-gui.cpp
	1. Configure the starting pattern by modifying `place_glider(dim, 10, 250, buf_current);` in main-gui.cpp
	1. Configure the speed by modifying `Sleep(100);` in main-gui.cpp
1. By implementing several variations as different structs, you can explore how optimisations affect performance.
	1. Implement a new struct with the same template arguments and update function
	1. Register it in main-benchmark.cpp by adding more lines similar to `run<SolverNaive<dim>>("naive");`
	1. Run life-benchmark to see the performance of each implementation

## Solver API

Indicate the api version by implementing this function in the solver class:
`constexpr static int get_api_version() { return 2;	}`
If this function is missing, the api version 1 is used.

### Version 1
* Class should be default constructible
* `void update(uint8_t* buf_current, uint8_t* buf_next)` should read from `buf_current` and write the next iteration to `buf_next`. Cells are represented as bytes, 1 = alive, 0 = dead. The buffer length is `dim * dim`. The buffers are swapped automatically outside of the solver class.

### Version 2
* Class should be default constructible
* `void init(uint8_t* buf)` should load the solver's internal state with the cells from `buf`. Cells in `buf` are represented as bytes, 1 = alive, 0 = dead. The buffer length is `dim * dim`. Internal representation is up to the solver.
* `void get_results(uint8_t* buf)` should write the solver's internal state to `buf`, same representation as above.
* `void update()` should advance the solver's internal state by one generation.