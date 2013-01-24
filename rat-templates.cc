#include <assert.h>
#include <utility>

#include "rat.hh"

using namespace std;

template <class NextHop>
void Rat::send( const unsigned int id, NextHop & next, const unsigned int tickno )
{
  assert( _packets_sent >= _packets_received );

  if ( _the_window == 0 ) {
    const Whisker & current_whisker( _whiskers.use_whisker( _memory, _track ) );
    _the_window = current_whisker.window( _the_window );
    _intersend_time = current_whisker.intersend();
  }

  while ( _packets_sent < _packets_received + _the_window ) {
    if ( _internal_tick > tickno + 1 ) {
      return;
    }

    Packet p( id, _packets_sent++, tickno );
    _memory.packet_sent( p );
    next.accept( move( p ) );
    _internal_tick += _intersend_time;
  }
}
