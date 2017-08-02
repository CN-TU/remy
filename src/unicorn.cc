#include <algorithm>
#include <limits>
#include <cassert>
#include "unicorn.hh"
#include "simulationresults.pb.h"
#include <cstdio>
#include <cmath>

#define LOSS_REWARD 0
#define INITIAL_WINDOW 5

using namespace std;

// FIXME: Don't create a thread in the Python code unless it is required.
Unicorn::Unicorn()
  :  _memory(),
     _packets_sent( 0 ),
     _packets_received( 0 ),
     _last_send_time( 0 ),
     _the_window( 0 ), // Start with the possibility to send at least one packet
     _intersend_time( 0 ),
     _flow_id( 0 ),
     _largest_ack( -1 ),
    //  _most_recent_ack ( -1 ),
     _thread_id(0),
     _rainbow(Rainbow::getInstance()),
    //  _outstanding_rewards()
    _put_actions(0),
    _put_rewards(0),
    _lost_since_last_time(0),
    _throughput_acc(0.0),
    _delay_acc(0.0),
    _last_packets_received_at_send_time(-1),
    _num_packets_received_from_send_time_acc(0),
    _receive_times_queue()
    // _sent_packets()
    // _sent_at_least_once(false)
{
  puts("Creating a Unicorn");
}

void Unicorn::packets_received( const vector< Packet > & packets ) {
  printf("~~~%lu: Oh my god, I received %lu packet!\n", _thread_id, packets.size());
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

    // printf("%lu: Lost rewards at received\n", _thread_id);
    _lost_since_last_time = packet.seq_num-_largest_ack-1;
    _memory.lost(_lost_since_last_time);
    put_lost_rewards(packet.packets_received_at_send_time-_last_packets_received_at_send_time-1);

    const double delay = packet.tick_received - packet.tick_sent;
    // printf("%lu: last_received:%f, received:%f\n", _thread_id, _memory._last_tick_received, packet.tick_received);
    const double throughput = 1.0;

    // printf("%lu: Calculated reward delay:%f, throughput:%f\n", _thread_id, packet_delay, throughput);

    if (packet.packets_received_at_send_time > _last_packets_received_at_send_time) {
      const double alpha = 10.0;
      const double beta = 10.0;

      const double start_time = _receive_times_queue.front();
      _receive_times_queue.pop();
      const double end_time = _receive_times_queue.front();
      const double throughput_final = alpha*_throughput_acc/(end_time-start_time);
      const double delay_final = beta*1.0/(_delay_acc/_num_packets_received_from_send_time_acc);
      printf("%lu: Calculated reward delay:%f, throughput:%f\n", _thread_id, delay_final, throughput_final);
      _rainbow.put_reward(_thread_id, throughput_final+delay_final);
      _put_rewards += 1;

      _num_packets_received_from_send_time_acc = 0;
      _throughput_acc = 0.0;
      _delay_acc = 0.0;
    }
    _throughput_acc += throughput;
    _delay_acc += delay;
    _num_packets_received_from_send_time_acc += 1;

    _last_packets_received_at_send_time = packet.packets_received_at_send_time;

    _packets_received += 1;
    vector<Packet> packet_for_memory_update;
    packet_for_memory_update.push_back(packet);
    _memory.packets_received( packet_for_memory_update, _flow_id, _largest_ack );
    // _sent_packets.erase(packet.seq_num);

    get_action(packet.tick_received, NULL);

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
  _the_window = 0; // Reset the window to 1
  _intersend_time = 0;
  _flow_id++;
  _largest_ack = _packets_sent - 1; /* Assume everything's been delivered */
  _put_actions = 0;
  _put_rewards = 0;
  _lost_since_last_time = 0;
  _throughput_acc = 0.0;
  _delay_acc = 0.0;
  _last_packets_received_at_send_time = -1;
  _num_packets_received_from_send_time_acc = 0;
  _receive_times_queue = queue<double>();
  assert( _flow_id != 0 );

  if (_thread_id == 0) {
    _thread_id = _rainbow.create_thread();
    printf("Assigned thread id %lu to Unicorn\n", _thread_id);
    // get_action();
  }
  printf("%lu: Starting\n", _thread_id);
  const double initial_window = INITIAL_WINDOW;
  get_action(0, &initial_window);

  /* initial window and intersend time */
  // const Whisker & current_whisker( _whiskers.use_whisker( _memory, _track ) );
  // _the_window = current_whisker.window( _the_window );
  // _intersend_time = current_whisker.intersend();
}

double Unicorn::next_event_time( const double & tickno ) const
{
  // return tickno;
  if ( int(_packets_sent) < _largest_ack + 1 + _the_window ) {
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

void Unicorn::get_action(const double& tickno, const double* action_to_put) {
  // action_struct action = _rainbow.get_action(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3), _memory.field(6), (double) _the_window/WINDOW_NORMALIZER});
  
  _receive_times_queue.push(tickno);
  
  action_struct* action_to_put_struct = NULL;
  if (action_to_put != NULL) {
    action_struct action_to_put_struct_temp = action_struct{*action_to_put, 1.0, 0.0};
    action_to_put_struct = &action_to_put_struct_temp;
  }

  action_struct action = _rainbow.get_action(
    _thread_id, 
    {
      // _memory.field(0), 
      // _memory.field(1), 
      // _memory.field(2), 
      // _memory.field(3), 
      _memory.field(6), // loss rate
      // (double) tickno - _last_send_time;
      (double) tickno - _memory._last_tick_sent, // time since last send
      (double) tickno - _memory._last_tick_received, // time since last receive
      (double) _lost_since_last_time, // losses since last receive
      _memory._send,
      _memory._rec
      // (tickno - _memory._last_tick_received)/LAST_SENT_TIME_NORMALIZER,
    },
    action_to_put_struct
  );
  // action.intersend /= 100.0;
  printf("%lu: action is: %f, %f, %f\n", _thread_id, action.window_increment, action.window_multiple, action.intersend);
  _put_actions += 1;

  _the_window = window(_the_window, action.window_increment, action.window_multiple);
  // _intersend_time = action.intersend;
}

void Unicorn::finish() {
  // const bool at_least_one_packet_sent = _put_actions>1;
  // if (!at_least_one_packet_sent) {
  //   _rainbow.put_reward(_thread_id, LOSS_REWARD);
  //   _put_rewards += 1;
  // }
  put_lost_rewards(_packets_sent-_largest_ack);
  // const bool at_least_one_packet_sent = true;
  printf("%lu: finish, _packets_sent: %u\n", _thread_id, _packets_sent);
  // _rainbow.finish(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3), _memory.field(6), (double) _the_window/WINDOW_NORMALIZER}, at_least_one_packet_sent);
  _rainbow.finish(_thread_id);
}

void Unicorn::put_lost_rewards(int number) {

  printf("%lu: Going to put loss rewards for %d intervals\n", _thread_id, number);
  for (int i=0; i<number; i++) {
    _rainbow.put_reward(_thread_id, LOSS_REWARD);
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
    printf("%lu: Lost rewards at destruction\n", _thread_id);
    // put_lost_rewards(_packets_sent-_largest_ack);
    // _rainbow.put_reward(_thread_id, LOSS_REWARD);
    // _put_rewards += 1;
    finish();
    _rainbow.delete_thread(_thread_id);
  }
}