#include <algorithm>
#include <limits>
#include <cassert>
#include "unicorn.hh"
#include "simulationresults.pb.h"
#include <cstdio>
#include <cmath>

#define LOSS_REWARD 0
// #define WINDOW_NORMALIZER (500.0)
#define INITIAL_WINDOW 0

using namespace std;

// FIXME: Don't create a thread in the Python code unless it is required.
Unicorn::Unicorn()
  :  _memory(),
     _packets_sent( 0 ),
     _packets_received( 0 ),
     _last_send_time( 0 ),
     _the_window( INITIAL_WINDOW ), // Start with the possibility to send at least one packet
     _intersend_time( 0 ),
     _flow_id( 0 ),
     _largest_ack( -1 ),
    //  _most_recent_ack ( -1 ),
     _thread_id(0),
     _unicorn_farm(UnicornFarm::getInstance()),
    //  _outstanding_rewards()
    _put_actions(0),
    _put_rewards(0)
    // _sent_packets()
    // _sent_at_least_once(false)
{
  puts("Creating a Unicorn");
}

void Unicorn::packets_received( const vector< Packet > & packets ) {
  printf("~~~%lu: Oh my god, I received %lu packets!\n", _thread_id, packets.size());
  // FIXME: Is this really super bad?
  assert(packets.size() == 1);

  // So this should only happen after a reset, when a packet arrives very late...
  if (_largest_ack >= packets.at( packets.size() - 1 ).seq_num) {
    printf("%lu: returning because _largest ack >= packet.seq_num\n", _thread_id);
    return;
  }

  const int previous_largest_ack = _largest_ack;

  for ( auto const &packet : packets ) {
    printf("%lu: packet.seq_num: %d, _largest_ack: %d\n", _thread_id, packet.seq_num, _largest_ack);

    puts("Lost rewards at received");
    _memory.lost(packet.seq_num-_largest_ack-1);
    put_lost_rewards(packet.seq_num-_largest_ack-1);

    const double alpha = 1000.0;
    const double beta = 100.0;
    const double packet_delay = 1.0/((packet.tick_received - packet.tick_sent)/alpha);
    // printf("%lu: last_received:%f, received:%f\n", _thread_id, _memory._last_tick_received, packet.tick_received);
    const double throughput = 1.0/((packet.tick_received-_memory._last_tick_received)/beta);
    const double reward = packet_delay+throughput;
    printf("%lu: Calculated reward delay:%f, throughput:%f\n", _thread_id, packet_delay, throughput);
    // if (reward < 0) {
    //   printf("delay: %f, throughput: %f\n", reward_delay, reward_throughput);
    // }
    // Ensure that the reward is never smaller or equal to the loss reward
    // assert(reward > 0);
    _unicorn_farm.put_reward(_thread_id, reward);
    _put_rewards += 1;

    _packets_received += 1;
    vector<Packet> packet_for_memory_update;
    packet_for_memory_update.push_back(packet);
    _memory.packets_received( packet_for_memory_update, _flow_id, _largest_ack );
    // _sent_packets.erase(packet.seq_num);

    // get_action();

    _largest_ack = packet.seq_num;
  }

  // FIXME: Why could _largest_ack already be larger than the current last packet?
  // _largest_ack = max( packets.at( packets.size() - 1 ).seq_num, _largest_ack );
  // FIXME: Apparently sometimes it happens... Shouldn't happen though...
  if (!(_largest_ack > previous_largest_ack)) {
    printf("%lu: largest ack: %d, previous largest ack: %d\n", _thread_id, _largest_ack, previous_largest_ack);
  }
  assert (_largest_ack > previous_largest_ack);
}

void Unicorn::reset( const double & )
{
  // assert(false);
  printf("%lu: Resetting\n", _thread_id);
  // _largest_ack -= 1;
  if (_thread_id > 0) {
    printf("%lu: Lost rewards at reset\n", _thread_id);
    put_lost_rewards(_packets_sent-_largest_ack);
    // _unicorn_farm.put_reward(_thread_id, LOSS_REWARD);
    // _put_rewards += 1;
    finish();
  }
  // if (_put_actions != _put_rewards) {
  //   printf("%lu: _put_actions: %lu, _put_rewards: %lu\n", _thread_id, _put_actions, _put_rewards);
  // }
  assert(_put_actions == _put_rewards);
  // assert(_sent_packets.size() == 0);

  _memory.reset();
  _last_send_time = 0;
  _the_window = INITIAL_WINDOW; // Reset the window to 1
  _intersend_time = 0;
  _flow_id++;
  _largest_ack = _packets_sent - 1; /* Assume everything's been delivered */
  _put_actions = 0;
  _put_rewards = 0;
  // _sent_packets.clear();
  assert( _flow_id != 0 );

  if (_thread_id == 0) {
    _thread_id = _unicorn_farm.create_thread();
    printf("Assigned thread id %lu to Unicorn\n", _thread_id);
    // get_action();
  }
  printf("%lu: Starting\n", _thread_id);
  get_action();

  /* initial window and intersend time */
  // const Whisker & current_whisker( _whiskers.use_whisker( _memory, _track ) );
  // _the_window = current_whisker.window( _the_window );
  // _intersend_time = current_whisker.intersend();
}

double Unicorn::next_event_time( const double & tickno ) const
{
  // return tickno;
  // if ( int(_packets_sent) < _largest_ack + 1 + _the_window ) {
    if ( _last_send_time + _intersend_time <= tickno ) {
      return tickno;
    } else {
      return _last_send_time + _intersend_time;
    }
  // } else {
  //   /* window is currently closed */
  //   return std::numeric_limits<double>::max();
  // }
}

void Unicorn::get_action() {
  // action_struct action = _unicorn_farm.get_action(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3), _memory.field(6), (double) _the_window/WINDOW_NORMALIZER});
  action_struct action = _unicorn_farm.get_action(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3), _memory.field(6)});
  printf("%lu: action is: %f, %f, %f\n", _thread_id, action.window_increment, action.window_multiple, action.intersend);
  _put_actions += 1;

  // _the_window = window(_the_window, action.window_increment, action.window_multiple);
  _intersend_time = action.intersend;
}

void Unicorn::finish() {
  const bool at_least_one_packet_sent = _put_actions>1;
  if (!at_least_one_packet_sent) {
    _unicorn_farm.put_reward(_thread_id, LOSS_REWARD);
    _put_rewards += 1;
  }
  // const bool at_least_one_packet_sent = true;
  printf("%lu: finish, _packets_sent: %u\n", _thread_id, _packets_sent);
  // _unicorn_farm.finish(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3), _memory.field(6), (double) _the_window/WINDOW_NORMALIZER}, at_least_one_packet_sent);
  _unicorn_farm.finish(_thread_id, at_least_one_packet_sent);
}

void Unicorn::put_lost_rewards(int number) {

  printf("%lu: Going to put %d lost packets\n", _thread_id, number);
  for (int i=0; i<number; i++) {
    _unicorn_farm.put_reward(_thread_id, LOSS_REWARD);
    _put_rewards += 1;

    // vector<Packet> packet_for_memory_update;
    // Packet packet = _sent_packets[i];
    // packet.lost = true;
    // packet_for_memory_update.push_back(packet);
    // _memory.packets_received( packet_for_memory_update, _flow_id, _largest_ack );
    // _sent_packets.erase(packet.seq_num);
  }
}

Unicorn::~Unicorn() {
  printf("Destroying Unicorn with thread id %lu\n", _thread_id);
  if (_thread_id > 0) {
    puts("Lost rewards at destruction");
    put_lost_rewards(_packets_sent-_largest_ack);
    // _unicorn_farm.put_reward(_thread_id, LOSS_REWARD);
    // _put_rewards += 1;
    finish();
    _unicorn_farm.delete_thread(_thread_id);
  }
}