#include <cassert>
#include <utility>

#include "rat.hh"

using namespace std;

template <class NextHop>
void Rat::send( const unsigned int id, NextHop & next, const double & tickno )
{
  assert( _packets_sent >= _packets_received );

  if ( _the_window == 0 ) {
    /* initial window and intersend time */
    const Whisker & current_whisker( _whiskers.use_whisker( _memory, _track ) );
    _the_window = current_whisker.window( _the_window );
    _intersend_time = current_whisker.intersend();
  }

  _whiskers.use_window( _the_window );

  while ( _packets_sent < _packets_received + _the_window ) {
    if ( _internal_tick > tickno ) {
      return;
    }

    Packet p( id, _flow_id, tickno );
    _packets_sent++;
    _memory.packet_sent( p );
    next.accept( move( p ) );
    _internal_tick += _intersend_time;
  }
}
