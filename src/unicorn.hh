#ifndef UNICORN_HH
#define UNICORN_HH

#include <vector>
#include <string>
#include <limits>
#include <queue>
#include <list>
#include <tuple>

#include "packet.hh"
#include "memory.hh"
#include "simulationresults.pb.h"

#include "rainbow.hh"
#include <cmath>

// #define MAX_WINDOW 1000000
#define MIN_WINDOW 1.0
#define MAX_WINDOW 10000.0

class Unicorn
{
private:

  Memory _memory;

  unsigned int _packets_sent, _packets_received;

  // bool _track;

  double _last_send_time;

  double _the_window;
  double _intersend_time;

  unsigned int _flow_id;
  int _largest_ack;

  long unsigned int _thread_id;
  Rainbow& _rainbow;
  // long unsigned int _previous_attempts;
  // long unsigned int _previous_attempts_acknowledged;
  // void put_lost_rewards();
  void get_action(const double& tickno);
  void finish();
  long unsigned int _put_actions;
  long unsigned int _put_rewards;

  //                   window        received      lost          start_t end_t   delay_acc last_packet_id
  std::list<std::tuple<unsigned int, unsigned int, unsigned int, double, double, double,   int>> _outstanding_rewards;

  // static double soft_ceil(const double x) {
  //   return MAX_WINDOW-log(1+exp(MAX_WINDOW-x));
  // }

  // static double soft_floor(const double x) {
  //   return log(1+exp(x+MIN_WINDOW))-MIN_WINDOW;
  // }

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
  double window(
    const double previous_window,  
    const double window_increment, 
    const double window_multiple
  ) const {
    double new_window = std::min(std::max(previous_window * window_multiple + window_increment, MIN_WINDOW), MAX_WINDOW);
    printf("%lu: new_window %f\n", _thread_id, new_window);
    return new_window;
  }
  // const double & intersend( void ) const { return _intersend; }
};

#endif
