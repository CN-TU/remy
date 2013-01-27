#ifndef EVALUATOR_HH
#define EVALUATOR_HH

#include <vector>

#include "random.hh"
#include "whiskertree.hh"
#include "network.hh"

class Evaluator
{
public:
  class Outcome
  {
  public:
    double score;
    std::vector< std::pair< NetConfig, std::vector< std::pair< double, double > > > > throughputs_delays;
    WhiskerTree used_whiskers;

    Outcome() : score( 0 ), throughputs_delays(), used_whiskers() {}
  };

  class ConfigRange
  {
  public:
    std::pair< double, double > link_packets_per_ms;
    std::pair< double, double > rtt_ms;
    unsigned int max_senders;
    bool lo_only;

    ConfigRange( void ) : link_packets_per_ms(), rtt_ms(), max_senders( 1 ), lo_only( false ) {}
  };

private:
  const PRNG _prng;
  const WhiskerTree _whiskers;

  std::vector< NetConfig > _configs;

public:
  Evaluator( const WhiskerTree & s_whiskers, const ConfigRange & range );
  Outcome score( const std::vector< Whisker > & replacements, const bool trace = false, const unsigned int carefulness = 1 ) const;
};

#endif
