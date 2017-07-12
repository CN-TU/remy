#ifndef UNICORN_HH
#define UNICORN_HH

#include <vector>
#include <string>
#include <limits>

#include "packet.hh"
#include "whiskertree.hh"
#include "memory.hh"
#include "simulationresults.pb.h"

class Unicorn
{
private:
  Memory _memory;

  unsigned int _packets_sent, _packets_received;

  // bool _track;

  double _last_send_time;

  int _the_window;
  double _intersend_time;

  unsigned int _flow_id;
  int _largest_ack;

  int _thread_id;

public:
  Unicorn();

  void packets_received( const std::vector< Packet > & packets );
  void reset( const double & tickno ); /* start new flow */

  template <class NextHop>
  void send( const unsigned int id, NextHop & next, const double & tickno,
	     const unsigned int packets_sent_cap = std::numeric_limits<unsigned int>::max() );

  Unicorn & operator=( const Unicorn & ) { assert( false ); return *this; }

  double next_event_time( const double & tickno ) const;

  const unsigned int & packets_sent( void ) const { return _packets_sent; }

  // SimulationResultBuffers::SenderState state_DNA() const;
  unsigned int window( const unsigned int previous_window ) const { return std::min( std::max( 0, int( previous_window * _window_multiple + _window_increment ) ), 1000000 ); }
  const double & intersend( void ) const { return _intersend; }
};

#endif
