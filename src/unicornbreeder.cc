#include <iostream>
#include <cstdio>

#include "unicornbreeder.hh"

using namespace std;

UnicornEvaluator::Outcome UnicornBreeder::run(const size_t iterations)
{
  for (size_t i=0; i<iterations; i++) {
    const UnicornEvaluator eval( _options.config_range );

    auto outcome(eval.score());

    const double final_score = outcome.score;
    printf("Finished one iteration in thread! Score is %f.\n", final_score);
    // TODO: Print score every now and then or generally do something...
  }

  const UnicornEvaluator eval2( _options.config_range );
  const auto new_score = eval2.score();

  return new_score;
}