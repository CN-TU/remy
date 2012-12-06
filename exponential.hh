#ifndef EXPONENTIAL_HH
#define EXPONENTIAL_HH

#include <boost/random/exponential_distribution.hpp>

#include "random.hh"

class Exponential
{
private:
  boost::random::exponential_distribution<> distribution;

public:
  Exponential( double rate ) : distribution( rate ) {}
  
  int sample( void ) { return distribution( get_generator() ); }
};

#endif
