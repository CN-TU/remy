#ifndef UNICORN_EVALUATOR_HH
#define UNICORN_EVALUATOR_HH

#include <string>
#include <vector>
#include <utility>

#include "random.hh"
#include "network.hh"
#include "unicorn.hh"
#include "configrange.hh"
#include "simulationresults.hh"

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
  unsigned int _prng_seed;
  // unsigned int _tick_count;

  std::vector< NetConfig > _configs;
  ConfigRangeUnicorn _config_range;
  size_t _thread_id;

  // ProblemBuffers::Problem _ProblemSettings_DNA ( void ) const;

public:
  UnicornEvaluator( const ConfigRangeUnicorn & range, const size_t thread_id );

  // ProblemBuffers::Problem DNA( const T & actions ) const;

  Outcome score() const;
  pair<UnicornEvaluator::Outcome, SimulationResultsUnicorn> score_with_logging() const;

  // static UnicornEvaluator::Outcome parse_problem_and_evaluate( const ProblemBuffers::Problem & problem );

  static Outcome score(
			const unsigned int prng_seed,
      const std::vector<NetConfig> & configs,
      const ConfigRangeUnicorn& config_range);

  static pair<UnicornEvaluator::Outcome, SimulationResultsUnicorn>
 score_with_logging(
			const unsigned int prng_seed,
      const std::vector<NetConfig> & configs,
      const ConfigRangeUnicorn& config_range);
};

#endif
