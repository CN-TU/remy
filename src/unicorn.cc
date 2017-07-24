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
     _the_window( 1 ), // Start with the possibility to send at least one packet
     _intersend_time( 0 ),
     _flow_id( 0 ),
     _largest_ack( -1 ),
    //  _most_recent_ack ( -1 ),
     _thread_id(0),
     _unicorn_farm(UnicornFarm::getInstance()),
    //  _outstanding_rewards()
    _put_actions(0),
    _put_rewards(0),
    _jitter_buffer()
    // _sent_at_least_once(false)
{
  puts("Creating a Unicorn");
}

// FIXME:FIXME:FIXME:IMMEDIATE: implement the function that gets called at reset.
// Implement a function that adapts the window.

void Unicorn::packets_received( const vector< Packet > & packets ) {
  printf("~~~%lu: Oh my god, I received %lu packets!\n", _thread_id, packets.size());
  // FIXME: Is this really super bad?
  assert(packets.size() <= 1);

  const int previous_largest_ack = _largest_ack;

  for ( auto &packet : packets ) {

    put_lost_rewards(packet.seq_num-_largest_ack-1, 0);

    _packets_received += 1;
    const vector<Packet> packet_for_memory_update;
    packet_for_memory_update.push_back(packet);
    _memory.packet_received( packet_for_memory_update, _flow_id, _largest_ack );

    const int packet_delay = packet.tick_received - packet.tick_sent;
    const double reward = 1.0/(packet_delay/multiplier);
    _unicorn_farm.put_reward(_thread_id, reward);
    _put_rewards += 1;

    get_action();

    _largest_ack = packet.seq_num;
  }

  put_contiguous_rewards(packet);

  // FIXME: Why could _largest_ack already be larger than the current last packet?
  // _largest_ack = max( packets.at( packets.size() - 1 ).seq_num, _largest_ack );
  assert (_largest_ack > previous_largest_ack);
}

void Unicorn::put_contiguous_rewards() {
  int previous_ack = _largest_ack;
  auto new_jitter_buffer = jitter_buffer;
  for (auto i=0; i<_jitter_buffer.size(); i++) {
    auto iterator = std::next(_jitter_buffer.begin(), i);
    auto current_packet = *iterator;
    if (current_packet.seq_num == previous_ack+1) {
      const double reward = Unicorn::calculate_reward(current_packet);
      printf("Putting reward %f\n", reward);
      _unicorn_farm.put_reward(_thread_id, reward);
      _put_rewards += 1;
      if (!current_packet.lost) {
        _packets_received += only_received_packets.size();
        /* Assumption: There is no reordering */
        _memory.packets_received( only_received_packets, _flow_id, _largest_ack );
      }
      new_jitter_buffer.erase(iterator);
    } else {
      break;
    }
    previous_ack = _jitter_buffer[i].seq_num;
  }
  _jitter_buffer = new_jitter_buffer;
}

const double Unicorn::calculate_reward(Packet p) const {
  const int packet_delay = packet.tick_received - packet.tick_sent;
  const double reward = 1.0/(packet_delay/multiplier);
  return reward;
}

void Unicorn::reset( const double & )
{
  // assert(false);
  puts("Resetting");
  // _largest_ack -= 1;
  if (_thread_id > 0) {
    put_missing_rewards(_packets_sent-_largest_ack);
    _unicorn_farm.finish(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3)});
  }
  assert(_put_actions == _put_rewards);

  _memory.reset();
  _last_send_time = 0;
  _the_window = 1; // Reset the window to 1
  _intersend_time = 0;
  _flow_id++;
  _largest_ack = _packets_sent - 1; /* Assume everything's been delivered */
  _put_actions = 0;
  _put_rewards = 0;
  _jitter_buffer.clear()
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

void Unicorn::get_action() {
  action_struct action = _unicorn_farm.get_action(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3)});
  _put_actions += 1;

  _the_window = window(_the_window, action.window_increment, action.window_multiple);
  _intersend_time = action.intersend;
}

void Unicorn::put_lost_rewards(size_t number) {
  assert(number >= 0);
  number = max(number, 0);

  printf("%lu: Going to put %lu lost packets\n", _thread_id, number);
  for (size_t i=0; i<number; i++) {
    _unicorn_farm.put_reward(_thread_id, LOSS_REWARD);
    _put_rewards += 1;
    get_action();
  }

}

Unicorn::~Unicorn() {
  printf("Destroying Unicorn with thread id %lu\n", _thread_id);
  if (_thread_id > 0) {
    _unicorn_farm.delete_thread(_thread_id);
  }
}