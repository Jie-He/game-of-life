#include <stdint.h>

template <int dim>
struct SolverNaive
{
	void update(uint8_t* buf_current, uint8_t* buf_next)
	{
		int index = 0;
		for (int col = 0; col < dim; col++)
		{
			for (int row = 0; row < dim; row++)
			{
				int n = 0;

				if (col - 1 >= 0)
				{
											n += buf_current[(col - 1) * dim + row	 ];
					if (row - 1 >= 0)		n += buf_current[(col - 1) * dim + row - 1];
					if (row + 1 < dim)		n += buf_current[(col - 1) * dim + row + 1];
				}

				if (col + 1 < dim)
				{
											n += buf_current[(col + 1) * dim + row	 ];
					if (row - 1 >= 0)		n += buf_current[(col + 1) * dim + row - 1];
					if (row + 1 < dim)		n += buf_current[(col + 1) * dim + row + 1];
				}

				if (row - 1 >= 0)			n += buf_current[col * dim + row - 1];
				if (row + 1 < dim)			n += buf_current[col * dim + row + 1];

				if (buf_current[index] == 0)
				{
					if (n == 3) buf_next[index] = 1;
					else		buf_next[index] = 0;
				}
				else
				{
					if (n < 2 || n > 3) buf_next[index] = 0;
					else				buf_next[index] = 1;
				}
				
				index += 1;
			}
		}
	}
};

template <int dim>
struct SolverBranchless
{
	void update(uint8_t* buf_current, uint8_t* buf_next)
	{
		for (int col = 1; col < dim-1; col++)
		{
			for (int row = 1; row < dim-1; row++)
			{
				int index = col * dim + row;
				int n = 0;

				n += buf_current[(col - 1) * dim + row    ];
				n += buf_current[(col - 1) * dim + row - 1];
				n += buf_current[(col - 1) * dim + row + 1];

				n += buf_current[(col + 1) * dim + row    ];
				n += buf_current[(col + 1) * dim + row - 1];
				n += buf_current[(col + 1) * dim + row + 1];

				n += buf_current[col * dim + row - 1];
				n += buf_current[col * dim + row + 1];

				if (buf_current[index] == 0)
				{
					if (n == 3)			buf_next[index] = 1;
					else				buf_next[index] = 0;
				}
				else
				{
					if (n < 2 || n > 3) buf_next[index] = 0;
					else				buf_next[index] = 1;
				}
			}
		}
	}
};

template <int dim>
struct SolverBranchlessLess
{
	void update(uint8_t* buf_current, uint8_t* buf_next)
	{
		for (int col = 1; col < dim - 1; col++)
		{
			for (int row = 1; row < dim - 1; row++)
			{
				int index = col * dim + row;
				uint8_t n = buf_current[(col - 1) * dim + row] + buf_current[(col - 1) * dim + row - 1] 
						  + buf_current[(col - 1) * dim + row + 1] + buf_current[(col + 1) * dim + row] 
						  + buf_current[(col + 1) * dim + row - 1] + buf_current[(col + 1) * dim + row + 1] 
					      + buf_current[col * dim + row - 1] + buf_current[col * dim + row + 1];
				buf_next[index] = ((n == 2) && buf_current[index]) || (n == 3);
			}
		}
	}
};

#include <immintrin.h>
template <int dim>
struct SolverAVX
{
	void update(uint8_t* buf_current, uint8_t* buf_next)
	{
		for (int row = 1; row < dim - 1; row++)
		{
			for (int col = 1; col < dim - 1; col+=32)
			{
				__m256i cc = _mm256_loadu_epi8(&buf_current[row * dim + col]);
				__m256i nn = _mm256_setzero_si256();
						nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row - 1) * dim + col - 1]));
						nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row - 1) * dim + col	 ]));
						nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row - 1) * dim + col + 1]));
						nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row	) * dim + col - 1]));
						nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row	) * dim + col + 1]));
						nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row + 1) * dim + col - 1]));
						nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row + 1) * dim + col	 ]));
						nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row + 1) * dim + col + 1]));

				__m256i	lc = _mm256_cmpeq_epi8(cc, m1);
				__m256i lf = _mm256_cmpeq_epi8(nn, m2);
				__m256i rh = _mm256_cmpeq_epi8(nn, m3); 
						
						cc = _mm256_or_si256(_mm256_and_si256(lf, lc), rh);
						cc = _mm256_and_si256(cc, m1);
				
#if (dim - 2) % 32 > 0
				if ((col + 32) - (dim - 1) > 0) 
						cc = _mm256_and_si256(cc, zeromask);		
#endif
				_mm256_storeu_epi8(&buf_next[row * dim + col], cc);
			}
		}
	}

	SolverAVX()
	{
		uint8_t mask[32];
		for (auto k = 0; k < 32; k++) 
			mask[k] = k < (dim - 2) % 32 ? 1 : 0;
		zeromask = _mm256_loadu_epi8(&mask[0]);
	}
	__m256i zeromask;
	__m256i m1 = _mm256_set1_epi8(1);
	__m256i m2 = _mm256_set1_epi8(2);
	__m256i m3 = _mm256_set1_epi8(3);
};

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
template <int dim, size_t tc, bool avx>
struct SolverMultiThread
{
	struct Worker
	{
		Worker(size_t pos, size_t rct, std::atomic<bool>& stop) : pos(pos), rct(rct), stop(stop)
		{
			thd = std::thread([this] { doTask(); });
		}

		~Worker() 
		{
			thd.join(); 
		}

		void setBuffers(uint8_t* buf_a, uint8_t* buf_b)
		{
			buf_current = buf_a;
			buf_next	= buf_b;
			ready		= true;
		}

		void doTask() 
		{
			// Waiting for ready
			while (!stop)
			{
				while (!ready && !stop){}
				ready = false;
				if (stop) return;
				
				if (avx)
					update_partial_avx();
				else
					update_partial_branchless();

				done = true;
			}
		}

		void update_partial_branchless()
		{
			for (auto row = pos; row < pos + rct; row++)
			{
				for (auto col = 1; col < dim - 1; col++)
				{
					auto index = row * dim + col;
					uint8_t n =	  buf_current[(row - 1) * dim + col] + buf_current[(row - 1) * dim + col - 1]
								+ buf_current[(row + 1) * dim + col - 1] + buf_current[(row + 1) * dim + col + 1]
								+ buf_current[(row - 1) * dim + col + 1] + buf_current[(row + 1) * dim + col]
								+ buf_current[row * dim + col - 1] + buf_current[row * dim + col + 1];
					buf_next[index] = ((n == 2) && buf_current[index]) || (n == 3);
				}
			}
		}

		void update_partial_avx()
		{
			for (auto row = pos; row < pos + rct; row++)
			{
				for (auto col = 1; col < dim - 1; col+=32)
				{
					__m256i cc = _mm256_loadu_epi8(&buf_current[row * dim + col]);
					__m256i nn = _mm256_setzero_si256();

					nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row - 1) * dim + col - 1]));
					nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row - 1) * dim + col    ]));
					nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row - 1) * dim + col + 1]));
					nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row	) * dim + col - 1]));
					nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row	) * dim + col + 1]));
					nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row + 1) * dim + col - 1]));
					nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row + 1) * dim + col    ]));
					nn = _mm256_add_epi8(nn, _mm256_loadu_epi8(&buf_current[(row + 1) * dim + col + 1]));

					__m256i	lc = _mm256_cmpeq_epi8(cc, m1);
					__m256i lf = _mm256_cmpeq_epi8(nn, m2);
					__m256i rh = _mm256_cmpeq_epi8(nn, m3);

					cc = _mm256_or_si256(_mm256_and_si256(lf, lc), rh);
					cc = _mm256_and_si256(cc, m1);

					_mm256_storeu_epi8(&buf_next[row * dim + col], cc);
				}
			}
		}

		std::thread thd;
		uint8_t *buf_current, *buf_next;
		size_t pos, rct;

		std::atomic<bool>& stop;
		std::atomic<bool> ready{ false };
		std::atomic<bool> done{ false };

		__m256i m1 = _mm256_set1_epi8(1);
		__m256i m2 = _mm256_set1_epi8(2);
		__m256i m3 = _mm256_set1_epi8(3);
	};

	SolverMultiThread()
	{
		size_t remainder = (dim - 2) % tc;
		size_t stepsize = (dim - 2) / tc;

		size_t row = 1;
		workers.reserve(tc);
		for (size_t i = 0; i < tc; i++)
		{
			workers[i] = new Worker(row, stepsize + (i < remainder ? 1 : 0), stop);
			row += stepsize + (i < remainder ? 1 : 0);
		}
	}

	~SolverMultiThread()
	{
		stop = true;
		for (int i = 0; i < tc; i++) 
			delete workers[i];
	}

	void update(uint8_t* buf_current, uint8_t* buf_next)
	{
		for (size_t i = 0; i < tc; i++) 
			workers[i]->setBuffers(buf_current, buf_next);

		bool notDone = true;
		while (notDone)
		{
			notDone = false;
			for (size_t i = 0; i < tc; i++)
			{
				if (!workers[i]->done)
				{
					notDone = true;
					break;
				}
			}
		}

		for (size_t i = 0; i < tc; i++)
			workers[i]->done = false;
	}

	std::atomic<bool> stop{ false };
	std::vector<Worker*> workers;
};