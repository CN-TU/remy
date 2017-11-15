#include <cassert>
#include <utility>

#include "unicorn.hh"

using namespace std;

template <class NextHop>
void Unicorn::send( const unsigned int id, NextHop & next, const double & tickno, const unsigned int packets_sent_cap )
{
  assert( int( _packets_sent ) >= _largest_ack + 1 );

  if ( _packets_sent >= packets_sent_cap ) {
    return;
  }

  if (
    (int(_packets_sent) < _largest_ack + 1 + _the_window ) &&
    (_packets_sent < packets_sent_cap)) {

    remy::Packet p( id, _flow_id, tickno, _packets_sent );
    _id_to_sent_during_flow[_packets_sent] = _flow_id;

    if (_packets_sent == packets_sent_cap-1) {
      p.last = true;
    }
    if (_last_send_time != 0) {
      _id_to_sent_during_action[_packets_sent] = _put_actions;
      _outstanding_rewards[_put_actions]["sent"] += 1;
      _outstanding_rewards[_put_actions]["intersend_duration_acc"] += tickno - _last_send_time;
    } else {
      p.first = true;
    }
    _packets_sent++;

    _memory.packet_sent( p );
    next.accept( p, tickno );

    _last_send_time = tickno;
    // printf("%lu: Sent packet\n", _thread_id);
  } else {
    // printf("%lu: Refrained from sending\n", _thread_id);
  }
}
