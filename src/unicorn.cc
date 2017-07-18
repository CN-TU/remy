#include <algorithm>
#include <limits>
#include <cassert>
#include "unicorn.hh"
#include "simulationresults.pb.h"
#include <cstdio>
#include <cmath>

using namespace std;

static double 

Unicorn::Unicorn()
  :  _memory(),
     _packets_sent( 0 ),
     _packets_received( 0 ),
     _last_send_time( 0 ),
     _the_window( 0 ),
     _intersend_time( 0 ),
     _flow_id( 0 ),
     _largest_ack( -1 )
{
  _unicorn_farm = UnicornFarm.getInstance();
  _thread_id = _unicorn_farm.create_thread();
}

void Unicorn::packets_received( const vector< Packet > & packets )
  if (packets.size() > 1) {
    printf("Received more than 1 packet: %d packets! That's super bad!\n", packets.size());
  }
  _packets_received += packets.size();
  /* Assumption: There is no reordering */
  _memory.packets_received( packets, _flow_id, _largest_ack );
  _largest_ack = max( packets.at( packets.size() - 1 ).seq_num, _largest_ack );

  for ( auto &packet : packets ) {
    packet_delay += packet.tick_received - packet.tick_sent;
  }

  for ( auto &packet : packets ) {
    packet_delay += packet.tick_received - packet.tick_sent;

    const double throughput_utility = 1;
    // FIXME: Find better solution for this
    const double multiplier = 0.01;
    // const double delay_penalty = log(packet_delay);
    const double delay_penalty = packet_delay);

    const double reward = throughput_utility - multiplier*delay_penalty;

    _unicorn_farm.put_reward(_thread_id, reward);
  }
}

void Unicorn::reset( const double & )
{
  _memory.reset();
  _last_send_time = 0;
  _the_window = 0;
  _intersend_time = 0;
  _flow_id++;
  _largest_ack = _packets_sent - 1; /* Assume everything's been delivered */
  assert( _flow_id != 0 );

  /* initial window and intersend time */
  const Whisker & current_whisker( _whiskers.use_whisker( _memory, _track ) );
  _the_window = current_whisker.window( _the_window );
  _intersend_time = current_whisker.intersend();
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

template <class NextHop>
void Unicorn::send( const unsigned int id, NextHop & next, const double & tickno,
		const unsigned int packets_sent_cap )
{
  assert( int( _packets_sent ) >= _largest_ack + 1 );

  action_struct action = _unicorn_farm.get_action(int thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3)});
  _the_window = window(_the_window, action.window_increment, action.window_multiple);
  _intersend_time = action.intersend;

  if ( (int( _packets_sent ) < _largest_ack + 1 + _the_window)
       and (_last_send_time + _intersend_time <= tickno) ) {

    // FIXME: Why exactly is that needed? Is it needed? Try removing it...
    /* Have we reached the end of the flow for now? */
    if ( _packets_sent >= packets_sent_cap ) {
      return;
    }

    Packet p( id, _flow_id, tickno, _packets_sent );
    _packets_sent++;
    _memory.packet_sent( p );
    next.accept( p, tickno );
    _last_send_time = tickno;
  }
}

SimulationResultBuffers::SenderState Unicorn::state_DNA() const
{
  SimulationResultBuffers::SenderState ret;
  ret.mutable_memory()->CopyFrom( _memory.DNA() );
  ret.set_window_size( _the_window );
  // ret.set_intersend_time( _intersend_time );
  return ret;
}