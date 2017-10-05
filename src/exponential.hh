#ifndef EXPONENTIAL_HH
#define EXPONENTIAL_HH

#include <random>

#include "random.hh"

class Exponential
{
private:
  std::exponential_distribution<> distribution;

public:
  const double rate;
  Exponential( const double & s_rate ) : distribution( s_rate ), rate(s_rate) {}

  void set_lambda( const double & rate )
  {
  	distribution = std::exponential_distribution<> ( rate );
  }

  double sample( PRNG & prng ) { return distribution( prng ); }
};

#endif
