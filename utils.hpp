#pragma once

#include <concepts>
#include <stdint.h>

void place_lidka(int dim, int x_pos, int y_pos, uint8_t* buf_current);
void place_glider(int dim, int x_pos, int y_pos, uint8_t* buf_current);

template <typename T> concept has_api_version_func = requires() { { T::get_api_version() } -> std::same_as<int>; };

template <typename Solver>
consteval int get_solver_api_version()
{
	if constexpr (has_api_version_func<Solver>)
		return Solver::get_api_version();
	else
		return 1;
}