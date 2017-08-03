#include <algorithm>
#include <limits>
#include <cassert>
#include "unicorn.hh"
#include "simulationresults.pb.h"
#include <cstdio>
#include <cmath>

using namespace std;

#define alpha 0.1
#define beta 0.1

Unicorn::Unicorn()
  : _memory(),
    _packets_sent( 0 ),
    _packets_received( 0 ),
    _last_send_time( 0 ),
    _the_window( MIN_WINDOW ), // Start with the possibility to send at least one packet
    _intersend_time( 0 ),
    _flow_id( 0 ),
    _largest_ack( -1 ),
    _thread_id(0),
    _rainbow(Rainbow::getInstance()),
    _put_actions(0),
    _put_rewards(0),
    _outstanding_rewards()
{
  puts("Creating a Unicorn");
}

void Unicorn::packets_received( const vector< Packet > & packets ) {
  printf("~~~%lu: Oh my god, I received %lu packet!\n", _thread_id, packets.size());
  assert(packets.size() == 1);

  // So this should only happen after a reset, when a packet arrives very late...
  if (_largest_ack >= packets.at( packets.size() - 1 ).seq_num) {
    printf("%lu: returning because _largest ack >= packet.seq_num\n", _thread_id);
    return;
  }

  const int previous_largest_ack = _largest_ack;

  for ( auto const &packet : packets ) {
    printf("%lu: packet.seq_num: %d, _largest_ack: %d\n", _thread_id, packet.seq_num, _largest_ack);
  
    const double delay = packet.tick_received - packet.tick_sent;
    const unsigned int lost_since_last_time = (unsigned int) packet.seq_num-_largest_ack-1;
    _memory.lost(lost_since_last_time);

    // Add new packets to tuples
    for (auto &tuple : _outstanding_rewards) {
      // printf("%lu: Seq num: %d, target num: %d\n", _thread_id, p.seq_num ,get<6>(tuple));
      // if (get<6>(tuple) == p.seq_num+2) {
      //   printf("%lu: Setting end_time to %f!\n", _thread_id, p.tick_sent);
      //   get<4>(tuple) = p.tick_sent;
      // }
      get<2>(tuple) += min(get<0>(tuple)-get<1>(tuple)-get<2>(tuple), lost_since_last_time);
      if (get<1>(tuple)+get<2>(tuple)<get<0>(tuple)) {
        get<5>(tuple) += delay;
        get<1>(tuple) += 1;
      }
    }

    // Check if tuple is full and if it is, put the corresponding reward for it
    for (auto &tuple : _outstanding_rewards) {
      if (get<0>(tuple) == get<1>(tuple) + get<2>(tuple)) {
        get<4>(tuple) = packet.tick_received;
        if (get<4>(tuple)-get<3>(tuple) == 0) {
          printf("%lu: get<4>(tuple): %f, get<3>(tuple): %f\n", _thread_id, get<4>(tuple), get<3>(tuple));
        }
        assert(get<4>(tuple)-get<3>(tuple) != 0);
        const double throughput_final = alpha*log(get<1>(tuple)/(get<4>(tuple)-get<3>(tuple)));
        const double delay_final = beta*log(get<5>(tuple)/get<1>(tuple));
        printf("%lu: Calculated reward delay:%f, throughput:%f\n", _thread_id, -delay_final, throughput_final);
        _rainbow.put_reward(_thread_id, throughput_final-delay_final);
        _put_rewards += 1;

      } else {
        break;
      }
    }

    // Remove the all the tuples for the beginning for which the reward has been accounted for
    auto i = _outstanding_rewards.begin();
    while (i != _outstanding_rewards.end()) {
      if (get<0>(*i) == get<1>(*i) + get<2>(*i))   {
        _outstanding_rewards.erase(i++);  // alternatively, i = items.erase(i);
      } else {
        break;
      }
    }

    _packets_received += 1;
    vector<Packet> packet_for_memory_update;
    packet_for_memory_update.push_back(packet);
    _memory.packets_received( packet_for_memory_update, _flow_id, _largest_ack );

    _largest_ack = packet.seq_num;

    get_action(packet.tick_received);
  }

  // FIXME: Why could _largest_ack already be larger than the current last packet?
  // _largest_ack = max( packets.at( packets.size() - 1 ).seq_num, _largest_ack );
  if (!(_largest_ack > previous_largest_ack)) {
    printf("%lu: largest ack: %d, previous largest ack: %d\n", _thread_id, _largest_ack, previous_largest_ack);
  }
  assert (_largest_ack > previous_largest_ack);
}

void Unicorn::reset( const double & tickno)
{
  // assert(false);
  printf("%lu: Resetting\n", _thread_id);
  // _largest_ack -= 1;
  if (_thread_id > 0) {
    printf("%lu: Lost rewards at reset\n", _thread_id);
    // put_lost_rewards(_packets_sent-_largest_ack);
    // _rainbow.put_reward(_thread_id, LOSS_REWARD);
    // _put_rewards += 1;
    finish();
  }
  if (_put_actions != _put_rewards) {
    printf("%lu: _put_actions: %lu, _put_rewards: %lu\n", _thread_id, _put_actions, _put_rewards);
  }
  assert(_put_actions == _put_rewards);
  // assert(_sent_packets.size() == 0);

  _memory.reset();
  _last_send_time = 0;
  _the_window = MIN_WINDOW; // Reset the window to 1
  _intersend_time = 0;
  _flow_id++;
  _largest_ack = _packets_sent - 1; /* Assume everything's been delivered */
  _put_actions = 0;
  _put_rewards = 0;
  _outstanding_rewards.clear();

  assert( _flow_id != 0 );

  if (_thread_id == 0) {
    _thread_id = _rainbow.create_thread();
    printf("Assigned thread id %lu to Unicorn\n", _thread_id);
    // get_action();
  }
  printf("%lu: Starting\n", _thread_id);
  get_action(tickno);

  /* initial window and intersend time */
  // const Whisker & current_whisker( _whiskers.use_whisker( _memory, _track ) );
  // _the_window = current_whisker.window( _the_window );
  // _intersend_time = current_whisker.intersend();
}

double Unicorn::next_event_time( const double & tickno ) const
{
  // return tickno;
  if ( int(_packets_sent) < _largest_ack + 1 + int(_the_window) ) {
    // if ( _last_send_time + _intersend_time <= tickno ) {
      return tickno;
    // } else {
      // return _last_send_time + _intersend_time;
    // }
  } else {
    /* window is currently closed */
    return std::numeric_limits<double>::max();
  }
}

void Unicorn::get_action(const double& tickno) {
  
  action_struct action = _rainbow.get_action(
    _thread_id, 
    {
      _memory.field(0),
      _memory.field(1), 
      _memory.field(2), 
      _memory.field(3), 
      _memory.field(6), // loss rate
      // (double) tickno - _memory._last_tick_sent, // time since last send
      // (double) tickno - _memory._last_tick_received, // time since last receive
      (double) _memory._lost_since_last_time, // losses since last receive
      _memory._send,
      _memory._rec
      // (tickno - _memory._last_tick_received)/LAST_SENT_TIME_NORMALIZER,
    }
  );
  // action.intersend /= 100.0;
  printf("%lu: action is: %f, %f, %f\n", _thread_id, action.window_increment, action.window_multiple, action.intersend);
  _put_actions += 1;

  _the_window = window(_the_window, action.window_increment, action.window_multiple);
  printf("%lu: target num: %d\n", _thread_id, _packets_sent + int(floor(_the_window)));
  _outstanding_rewards.push_back({int(floor(_the_window)), 0, 0, tickno, -1.0, 0.0, _packets_sent + int(floor(_the_window))});
}

void Unicorn::finish() {
  // const bool at_least_one_packet_sent = _put_actions>1;
  // if (!at_least_one_packet_sent) {
  //   _rainbow.put_reward(_thread_id, LOSS_REWARD);
  //   _put_rewards += 1;
  // }
  put_lost_rewards();
  // const bool at_least_one_packet_sent = true;
  printf("%lu: finish, _packets_sent: %u\n", _thread_id, _packets_sent);
  // _rainbow.finish(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3), _memory.field(6), (double) _the_window/WINDOW_NORMALIZER}, at_least_one_packet_sent);
  _rainbow.finish(_thread_id);
}

void Unicorn::put_lost_rewards() {
  // printf("%lu: Going to put loss rewards for %d intervals\n", _thread_id, number);
  for (auto &tuple : _outstanding_rewards) {
    get<2>(tuple) = get<0>(tuple) - get<1>(tuple);
    const double throughput_final = alpha*log(get<1>(tuple)/(get<4>(tuple)-get<3>(tuple)));
    const double delay_final = beta*log(get<5>(tuple)/get<1>(tuple));
    printf("%lu: Calculated reward delay:%f, throughput:%f\n", _thread_id, -delay_final, throughput_final);
    _rainbow.put_reward(_thread_id, throughput_final-delay_final);
    _put_rewards += 1;
  }
}

Unicorn::~Unicorn() {
  printf("Destroying Unicorn with thread id %lu\n", _thread_id);
  if (_thread_id > 0) {
    printf("%lu: Lost rewards at destruction\n", _thread_id);
    // put_lost_rewards(_packets_sent-_largest_ack);
    // _rainbow.put_reward(_thread_id, LOSS_REWARD);
    // _put_rewards += 1;
    finish();
    _rainbow.delete_thread(_thread_id);
  }
}