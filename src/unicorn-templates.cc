#include <cassert>
#include <utility>
#include <cmath>

#include "unicorn.hh"

using namespace std;

template <class NextHop>
void Unicorn::send( const unsigned int id, NextHop & next, const double & tickno, const unsigned int packets_sent_cap )
{
  assert( int( _packets_sent ) >= _largest_ack + 1 );

  if ( _packets_sent >= packets_sent_cap ) {
    return;
  }

  if (_last_send_time > 0 && tickno - _last_send_time >= TIMEOUT_THRESHOLD) {
    printf("%lu: timeout occurred! tickno: %f, _last_send_time: %f\n", _thread_id, tickno, _last_send_time);
    reset(tickno);
  }

  // printf("left:%d, right:%d\n",int(_packets_sent),  (int) _largest_ack + 1 + (int) floor(_the_window));
  if (
    (int(_packets_sent) < _largest_ack + 1 + _the_window ) &&
    (_packets_sent < packets_sent_cap)) {

    remy::Packet p( id, _flow_id, tickno, _packets_sent );
    _id_to_sent_during_flow[_packets_sent] = _flow_id;

    if (_packets_sent == packets_sent_cap-1) {
      _active_flows.erase(_flow_id);
    }
    if (_last_send_time == 0) {
      _memory._last_tick_sent = tickno;
      _memory._last_tick_received = tickno;
      _memory._lost_since_last_time = 0;
      _flow_to_last_received[_flow_id] = tickno;
    }

    _id_to_sent_during_action[_packets_sent] = _put_actions;
    _outstanding_rewards[_put_actions]["sent"] += 1;
    _outstanding_rewards[_put_actions]["intersend_duration_acc"] += tickno - _last_send_time;
    _packets_sent++;

    _last_send_time = tickno;

    _memory.packet_sent( p );
    next.accept( p, tickno );
  }
}
