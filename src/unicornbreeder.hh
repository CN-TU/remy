#ifndef UNICORNBREEDER_HH
#define UNICORNBREEDER_HH

// #include "breeder.hh"
#include "configrange.hh"
#include "unicornevaluator.hh"

struct BreederOptionsUnicorn
{
  ConfigRangeUnicorn config_range = ConfigRangeUnicorn();
};

class UnicornBreeder
{
private:
  BreederOptionsUnicorn _options;
  const size_t _thread_id;

public:
  UnicornBreeder( const BreederOptionsUnicorn & s_options, const size_t thread_id)
  : _options( s_options ),
  _thread_id(thread_id)
  {};

  void run(const size_t iterations);
};

#endif