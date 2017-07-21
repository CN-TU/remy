#include <algorithm>
#include <limits>
#include <cassert>
#include "unicorn.hh"
#include "simulationresults.pb.h"
#include <cstdio>
#include <cmath>

#define LOSS_REWARD 0

using namespace std;

// FIXME: Don't create a thread in the Python code unless it is required. 
Unicorn::Unicorn()
  :  _memory(),
     _packets_sent( 0 ),
     _packets_received( 0 ),
     _last_send_time( 0 ),
     _the_window( 0 ),
     _intersend_time( 0 ),
     _flow_id( 0 ),
     _largest_ack( -1 ),
    //  _most_recent_ack ( -1 ),
     _thread_id(0),
     _unicorn_farm(UnicornFarm::getInstance())
{
  puts("Creating a Unicorn");
}

void Unicorn::packets_received( const vector< Packet > & packets ) {
  // FIXME: Is this really super bad?
  if (packets.size() > 1) {
    printf("Received more than 1 packet: %d packets! That's super bad!\n", (unsigned int) packets.size());
  }
  _packets_received += packets.size();
  /* Assumption: There is no reordering */
  _memory.packets_received( packets, _flow_id, _largest_ack );
  int previous_largest_ack = _largest_ack;

  // for ( auto &packet : packets ) {
  //   packet_delay += packet.tick_received - packet.tick_sent;
  // }

  for ( auto &packet : packets ) {

    const size_t packets_missing = packet.seq_num - _largest_ack - 1;

    for (size_t i=0; i<packets_missing; i++) {
      _unicorn_farm.put_reward(_thread_id, LOSS_REWARD);
    }

    const int packet_delay = packet.tick_received - packet.tick_sent;

    const double throughput_utility = 1;
    // FIXME: Find better solution for this
    const double multiplier = 0.01;
    // const double delay_penalty = log(packet_delay);
    const double delay_penalty = (double) packet_delay;

    const double reward = throughput_utility - multiplier*delay_penalty;
    // FIXME: Whether the reward is positive doesn't matter I guess?
    // assert(reward > 0.0);

    _unicorn_farm.put_reward(_thread_id, reward);

    _largest_ack = packet.seq_num;
  }

  // FIXME: Why could _largest_ack already be larger than the current last packet?
  // _largest_ack = max( packets.at( packets.size() - 1 ).seq_num, _largest_ack );
  assert (_largest_ack > previous_largest_ack);
}

void Unicorn::reset( const double & )
{

  put_missing_rewards();
  _memory.reset();
  _last_send_time = 0;
  _the_window = 0;
  _intersend_time = 0;
  _flow_id++;
  _largest_ack = _packets_sent - 1; /* Assume everything's been delivered */
  assert( _flow_id != 0 );

  /* initial window and intersend time */
  // const Whisker & current_whisker( _whiskers.use_whisker( _memory, _track ) );
  // _the_window = current_whisker.window( _the_window );
  // _intersend_time = current_whisker.intersend();
}

double Unicorn::next_event_time( const double & tickno ) const
{
  if ( int(_packets_sent) < _largest_ack + 1 + _the_window ) {
    if ( _last_send_time + _intersend_time <= tickno ) {
      return tickno;
    } else {
      return _last_send_time + _intersend_time;
    }
  } else {
    /* window is currently closed */
    return std::numeric_limits<double>::max();
  }
}

// SimulationResultBuffers::SenderState Unicorn::state_DNA() const
// {
//   SimulationResultBuffers::SenderState ret;
//   ret.mutable_memory()->CopyFrom( _memory.DNA() );
//   ret.set_window_size( _the_window );
//   // ret.set_intersend_time( _intersend_time );
//   return ret;
// }

void Unicorn::put_missing_rewards() {
  if (_packets_sent > 0) {
    printf("Unicorn sent %i\n", _packets_sent);
    // When everything is finished and packets were lost in the end, put reward 0 for each of them. 
    for (size_t i=0; i<(_packets_sent-_largest_ack-1); i++) {
      _unicorn_farm.put_reward(_thread_id, LOSS_REWARD);
    }
    _unicorn_farm.finish(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3)});
  }
}

Unicorn::~Unicorn() {
  printf("Destroying Unicorn with thread id %lu\n", _thread_id);
  put_missing_rewards();
  if (_thread_id > 0) {
    _unicorn_farm.delete_thread(_thread_id);
  }
}