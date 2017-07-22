#ifndef UNICORN_HH
#define UNICORN_HH

#include <vector>
#include <string>
#include <limits>

#include "packet.hh"
#include "memory.hh"
#include "simulationresults.pb.h"

#include "unicornfarm.hh"

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

  long unsigned int _thread_id;
  UnicornFarm& _unicorn_farm;
  long unsigned int _previous_attempts;
  long unsigned int _previous_attempts_acknowledged;
  void put_missing_rewards(const int seq_num, const int previous_attempts);
  long unsigned int _put_actions;
  long unsigned int _put_rewards;

  // bool _sent_at_least_once;

public:
  Unicorn();
  ~Unicorn();

  void packets_received( const std::vector< Packet > & packets );
  void reset( const double & tickno ); /* start new flow */

  template <class NextHop>
  void send( const unsigned int id, NextHop & next, const double & tickno,
	     const unsigned int packets_sent_cap = std::numeric_limits<unsigned int>::max() );

  // Unicorn & operator=( const Unicorn & ) { assert( false ); return *this; }
	// Unicorn(Unicorn const&) = delete;
	void operator=(Unicorn const&) = delete;

  double next_event_time( const double & tickno ) const;

  const unsigned int & packets_sent( void ) const { return _packets_sent; }

  // SimulationResultBuffers::SenderState state_DNA() const;
  unsigned int window(
    const unsigned int previous_window,  
    const int window_increment, 
    const double window_multiple
  ) const {
    unsigned int new_window = std::min( std::max( 0, int( previous_window * window_multiple + window_increment ) ), 1000000 );
    printf("new_window %u\n", new_window);
    return new_window;
  }
  // const double & intersend( void ) const { return _intersend; }
};

#endif
