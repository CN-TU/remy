#include <algorithm>
#include <limits>
#include <cassert>
#include <utility>

#include "unicorn.hh"

using namespace std;

Unicorn::Unicorn()
  :  _memory(),
     _packets_sent( 0 ),
     _packets_received( 0 ),
     _last_send_time( 0 ),
     _the_window( 0 ),
     _intersend_time( 0 ),
     _flow_id( 0 ),
     _largest_ack( -1 ),
     _thread_id( std::numeric_limits<int>::min(); )
{
}

void Unicorn::packets_received( const vector< Packet > & packets ) {
  _packets_received += packets.size();
  /* Assumption: There is no reordering */
  _memory.packets_received( packets, _flow_id, _largest_ack );
  _largest_ack = max( packets.at( packets.size() - 1 ).seq_num, _largest_ack );

  // TODO: call function for received packet and for reward once for
  // EACH packet received

  // TODO: do that for each packet too!
  _the_window = window( _the_window, window_increment, window_multiple );
  _intersend_time = intersend();
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

  if ( _the_window == 0 ) {
    /* initial window and intersend time */
    const Whisker & current_whisker( _whiskers.use_whisker( _memory, _track ) );
    _the_window = current_whisker.window( _the_window );
    _intersend_time = current_whisker.intersend();
  }

  if ( (int( _packets_sent ) < _largest_ack + 1 + _the_window)
       and (_last_send_time + _intersend_time <= tickno) ) {

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
  ret.set_intersend_time( _intersend_time );
  return ret;
}