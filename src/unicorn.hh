#ifndef UNICORN_HH
#define UNICORN_HH

#include <vector>
#include <string>
#include <limits>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include "packet.hh"
#include "memory.hh"

#include "rainbow.hh"
#include <cmath>

// #include "simulationresults.hh"

#include "../protobufs/simulationresults.pb.h"

// #define MIN_WINDOW_UNICORN 1.0
// #define MAX_WINDOW_UNICORN 500.0

#define TIMEOUT_THRESHOLD 1000.0

class Unicorn
{
protected:
  Memory _memory;

  long unsigned int _packets_sent, _packets_received, _packets_lost;
  double _last_send_time;
  double _the_window;

  unsigned int _flow_id;
  long int _largest_ack;

  long unsigned int _thread_id;
  Rainbow& _rainbow;
  // long unsigned int _previous_attempts;
  // long unsigned int _previous_attempts_acknowledged;
  // void put_lost_rewards();
  void get_action(const double& tickno);
  void finishFlow();
  long unsigned int _put_actions;
  long unsigned int _put_rewards;

  // std::list<std::unordered_map<std::string, double>> _outstanding_rewards;
  std::map<uint32_t, std::unordered_map<std::string, double>> _outstanding_rewards;
  // std::unordered_map<uint32t_t, uint32_t> _packet_to_action;
  double _start_tick;
  bool _training;
  double _delay_delta;

  std::map<long unsigned int, long unsigned int> _id_to_sent_during_action;
  std::map<long unsigned int, unsigned int> _id_to_sent_during_flow;
  std::map<long unsigned int, double> _flow_to_last_received;
  std::unordered_set<unsigned int> _active_flows;

  // static double soft_ceil(const double x) {
  //   return MAX_WINDOW-log(1+exp(MAX_WINDOW-x));
  // }

  // static double soft_floor(const double x) {
  //   return log(1+exp(x+MIN_WINDOW))-MIN_WINDOW;
  // }

public:
  Unicorn(const bool& cooperative, const double& delay_delta = 1);
  ~Unicorn();

  bool packets_lost( const std::vector< remy::Packet > & packets );
  void packets_received( const std::vector< remy::Packet > & packets );
  void reset( const double & tickno ); /* start new flow */

  template <class NextHop>
  void send( const unsigned int id, NextHop & next, const double & tickno,
	     const unsigned int packets_sent_cap = std::numeric_limits<unsigned int>::max() );

  // Unicorn & operator=( const Unicorn & ) { assert( false ); return *this; }
	// Unicorn(Unicorn const&) = delete;
	void operator=(Unicorn const&) = delete;

  double next_event_time( const double & tickno ) const;

  const long unsigned int & packets_sent( void ) const { return _packets_sent; }

  SimulationResultBuffers::SenderState state_DNA() const;
};

#endif
