#include <chrono>
#include <optional>
#include <algorithm>

#include "solver.hpp"
#include "reference.hpp"
#include "utils.hpp"

constexpr int dim = 4098; //2050;
uint8_t* buf_current;
uint8_t* buf_next;

uint32_t count_alive()
{
	uint32_t alive = 0;
	for (int i = 0; i < dim * dim; i++)
		alive += buf_current[i];

	return alive;
}

template <typename Solver>
int64_t run(const char* name, std::optional<int64_t> ref_timing = std::nullopt)
{
	int64_t best_time = std::numeric_limits<int64_t>::max();
	int num_runs = 0;

	while (num_runs < 3 || (num_runs < 10 && best_time < 500))
	{
		for (int i = 0; i < dim * dim; i++)
			buf_current[i] = 0;

	    for (int i = 0; i < dim * dim; i++)
			buf_next[i] = 0;

	    place_lidka(dim, 1000, 1000, buf_current);

	    auto start_timepoint = std::chrono::high_resolution_clock::now();

	    Solver solver;

	    for (int gen = 0; gen < 100; gen++)
	    {
			solver.update(buf_current, buf_next);
			std::swap(buf_current, buf_next);
	    }

	    auto end_timepoint = std::chrono::high_resolution_clock::now();

		auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_timepoint - start_timepoint).count();

		best_time = std::min(best_time, duration_ms);
		num_runs += 1;
	}

    auto alive = count_alive();
    
    if (ref_timing.has_value())
    {
		float speedup = (float)ref_timing.value() / (float)best_time;
		printf("%s: %zd ms, alive=%d, runs=%d, speedup=%.1fx\n", name, best_time, alive, num_runs, speedup);
    }
    else
		printf("%s: %zd ms, alive=%d, runs=%d\n", name, best_time, alive, num_runs);

    return best_time;
}

int main()
{
	buf_current = new uint8_t[dim * dim];
	buf_next = new uint8_t[dim * dim];

	auto ref_timing = run<SolverReference<dim>>("reference");
	constexpr bool use_avx = true;
	run<SolverMultiThread<dim,  1, use_avx>>("MultiThread  1", ref_timing);
	run<SolverMultiThread<dim,  2, use_avx>>("MultiThread  2", ref_timing);
	run<SolverMultiThread<dim,  4, use_avx>>("MultiThread  4", ref_timing);
	run<SolverMultiThread<dim,  8, use_avx>>("MultiThread  8", ref_timing);
	run<SolverMultiThread<dim, 16, use_avx>>("MultiThread 16", ref_timing);
	run<SolverMultiThread<dim, 20, use_avx>>("MultiThread 20", ref_timing);
	run<SolverMultiThread<dim, 22, use_avx>>("MultiThread 22", ref_timing);
	run<SolverMultiThread<dim, 24, use_avx>>("MultiThread 24", ref_timing);
	run<SolverAVX<dim>>						("           AVX", ref_timing);
	run<SolverBranchlessLess<dim>>			("branchlessLess", ref_timing);
	run<SolverBranchless<dim>>				("    branchless", ref_timing);
	run<SolverNaive<dim>>					("       mynaive", ref_timing);
}
