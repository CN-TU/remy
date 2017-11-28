#ifndef RANDOM_HH
#define RANDOM_HH

#include <random>

typedef std::default_random_engine PRNG;

extern PRNG & global_PRNG();
extern PRNG* global_PRNG(const size_t thread_id
);

#endif
