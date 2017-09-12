#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "unicornbreeder.hh"

using namespace std;

UnicornEvaluator::Outcome UnicornBreeder::run(const size_t iterations)
{
  size_t iteration_number = 0;
  char file_name[64 * sizeof(char)];
  sprintf(file_name, "stats/thread%lu", _thread_id);
  FILE* f = fopen(file_name, "w");
  if (f == NULL) {
    puts("Error opening file!\n");
    exit(1);
  }
  for (size_t i=0; i<iterations; i++) {
    const UnicornEvaluator eval( _options.config_range );

    auto outcome(eval.score());

    const double final_score = outcome.score;
    fprintf(f, "%lu,%lu,%f\n", iteration_number, (unsigned long)time(NULL), final_score);
    fflush(f);
    printf("Finished one iteration (%lu) in thread %lu! Score is %f.\n", iteration_number, _thread_id, final_score);

    iteration_number += 1;
  }

  fclose(f);

  const UnicornEvaluator eval2( _options.config_range );
  const auto new_score = eval2.score();

  return new_score;
}