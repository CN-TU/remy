#ifndef UNICORN_EVALUATOR_HH
#define UNICORN_EVALUATOR_HH

#include <string>
#include <vector>

#include "random.hh"
#include "network.hh"
#include "unicorn.hh"
// #include "problem.pb.h"
// #include "answer.pb.h"

class UnicornEvaluator
{
public:
  class Outcome
  {
  public:
    double score;
    std::vector< std::pair< NetConfig, std::vector< std::pair< double, double > > > > throughputs_delays;

    Outcome() : score( 0 ), throughputs_delays() {}

    // Outcome( const AnswerBuffers::Outcome & dna );

    // AnswerBuffers::Outcome DNA( void ) const;
  };

private:
  const unsigned int _prng_seed;
  // unsigned int _tick_count;

  std::vector< NetConfig > _configs;

  // ProblemBuffers::Problem _ProblemSettings_DNA ( void ) const;

public:
  UnicornEvaluator( const ConfigRangeUnicorn & range );
  
  // ProblemBuffers::Problem DNA( const T & actions ) const;

  Outcome score() const;

  // static UnicornEvaluator::Outcome parse_problem_and_evaluate( const ProblemBuffers::Problem & problem );

  static Outcome score(
			const unsigned int prng_seed,
			const std::vector<NetConfig> & configs);
};

#endif
