#ifndef RECEIVER_HH
#define RECEIVER_HH

#include <vector>

#include "packet.hh"

class Receiver
{
private:
  std::vector< std::vector< Packet > > _collector;

public:
  Receiver( const int num_senders );

  void accept( const Packet & p, const int tickno );
  std::vector< Packet > collect( const unsigned int src );
};

#endif
