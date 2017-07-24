#include <cassert>
#include <utility>

#include "unicorn.hh"

using namespace std;

template <class NextHop>
void Unicorn::send( const unsigned int id, NextHop & next, const double & tickno,
		const unsigned int packets_sent_cap )
{
  // _sent_at_least_once = true;
  if (_thread_id == 0) {
    _thread_id = _unicorn_farm.create_thread();
    printf("Assigned thread id %lu to Unicorn\n", _thread_id);

    assert(_memory.field(0) == _memory.field(1));
    action_struct action = _unicorn_farm.get_action(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3)});
    _put_actions += 1;

    _the_window = window(_the_window, action.window_increment, action.window_multiple);
    _intersend_time = action.intersend;
  }
  assert( int( _packets_sent ) >= _largest_ack + 1 );

  if ( _packets_sent >= packets_sent_cap ) {
    return;
  }

  // printf("Memory fields: %f,%f,%f,%f\n", _memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3));

  if ( ((int( _packets_sent ) < _largest_ack + 1 + _the_window)
       and (_last_send_time + _intersend_time <= tickno)) and (_packets_sent < packets_sent_cap) ) {
    // FIXME: Why exactly is that needed? Is it needed? Try removing it...
    /* Have we reached the end of the flow for now? */

    Packet p( id, _flow_id, tickno, _packets_sent);

    _packets_sent++;

    _memory.packet_sent( p );
    next.accept( p, tickno );
    _last_send_time = tickno;
    printf("%lu: Sent packet\n", _thread_id);
  } else {
    printf("%lu: Refrained from sending\n", _thread_id);
  }
}