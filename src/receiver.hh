#ifndef RECEIVER_HH
#define RECEIVER_HH

#include <vector>
#include <cassert>

#include "packet.hh"
using namespace remy;

class Receiver
{
private:
  std::vector< std::vector< Packet > > _collector;
  // std::vector< std::vector< Packet > > _collector_lost;
  void autosize( const unsigned int index );
  // void autosize_lost( const unsigned int index );

public:
  Receiver();

  void accept( const Packet & p, const double & tickno ) noexcept;
  // void accept_lost( Packet & p, const double & tickno ) noexcept;
  const std::vector< Packet > & packets_for( const unsigned int src ) {
    return _collector[ src ];
  }
  // const std::vector<Packet>* all_packets_for( const unsigned int src ) {
  //   std::vector<Packet>* concat = new std::vector<Packet>();
  //   concat->reserve( _collector[ src ].size() + _collector_lost[ src ].size() ); // preallocate memory
  //   concat->insert( concat->end(), _collector[ src ].begin(), _collector[ src ].end() );
  //   concat->insert( concat->end(), _collector_lost[ src ].begin(), _collector_lost[ src ].end() );
  //   // FIXME: If the size were ever larger than 1, it would need to be sorted.
  //   assert(concat->size()<=1);
  //   return concat;
  // }
  void clear( const unsigned int src ) { _collector[ src ].clear(); }
  bool readable( const unsigned int src ) const noexcept
  { return (src < _collector.size()) && (!_collector[ src ].empty()); }

  double next_event_time( const double & tickno ) const;
};

#endif
