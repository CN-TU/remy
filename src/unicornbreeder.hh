#ifndef UNICORNBREEDER_HH
#define UNICORNBREEDER_HH

#include "breeder.hh"
#include "unicornevaluator.hh"

class UnicornBreeder
{
private:
  BreederOptionsUnicorn _options;

public:
  UnicornBreeder( const BreederOptionsUnicorn & s_options) 
  : _options( s_options )
  {};

  UnicornEvaluator::Outcome run(const size_t iterations);
};

#endif