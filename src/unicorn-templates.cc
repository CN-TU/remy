#include <cassert>
#include <utility>

#include "unicorn.hh"

using namespace std;

template <class NextHop>
void Unicorn::send( const unsigned int id, NextHop & next, const double & tickno,
		const unsigned int packets_sent_cap )
{
  // _sent_at_least_once = true;

  assert( int( _packets_sent ) >= _largest_ack + 1 );

  if ( _packets_sent >= packets_sent_cap ) {
    return;
  }

  // printf("Memory fields: %f,%f,%f,%f\n", _memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3));

  if (
    // (int( _packets_sent ) < _largest_ack + 1 + _the_window) && 
    (_last_send_time + _intersend_time <= tickno) && 
    (_packets_sent < packets_sent_cap)) {
    // FIXME: Why exactly is that needed? Is it needed? Try removing it...
    /* Have we reached the end of the flow for now? */

    Packet p( id, _flow_id, tickno, _packets_sent);
    // _sent_packets.insert({_packets_sent, p});
    _packets_sent++;

    _memory.packet_sent( p );
    next.accept( p, tickno );
    _last_send_time = tickno;
    get_action();
    printf("%lu: Sent packet\n", _thread_id);
  } else {
    printf("%lu: Refrained from sending\n", _thread_id);
  }
}