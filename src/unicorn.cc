#include <algorithm>
#include <limits>
#include <cassert>
#include "unicorn.hh"
#include <cstdio>
#include <cmath>

using namespace std;

Unicorn::Unicorn(const bool& cooperative, const double& delay_delta)
  : _memory(),
    _packets_sent( 0 ),
    _packets_received( 0 ),
    _packets_lost( 0 ),
    _last_send_time( 0 ),
    _the_window( MIN_WINDOW_UNICORN ), // Start with the possibility to send at least one packet
    _flow_id( 0 ),
    _largest_ack( -1 ),
    _thread_id(0),
    _rainbow(Rainbow::getInstance(cooperative)),
    _put_actions(0),
    _put_rewards(0),
    _outstanding_rewards(),
    _start_tick(0.0),
    _training(true),
    _delay_delta(delay_delta),
    _id_to_sent_during_action(),
    _id_to_sent_during_flow(),
    _flow_to_last_received(),
    _active_flows()
{
  // puts("Creating a Unicorn");
}

void Unicorn::packets_received( const vector< remy::Packet > & packets ) {
  // printf("~~~%lu: Oh my god, I received %lu packet!\n", _thread_id, packets.size());
  assert(packets.size() == 1);

  // So this should only happen after a reset, when a packet arrives very late...
  // assert (_largest_ack < packets.at( packets.size() - 1 ).seq_num);
  // const int previous_largest_ack = _largest_ack;

  // _largest_ack = max( (int) packets.at( packets.size() - 1 ).seq_num, _largest_ack );

  for ( auto const &packet : packets ) {
    // if (_id_to_sent_during_flow[packet.seq_num] != _flow_id) {
    //   continue;
    // }

    if (_id_to_sent_during_flow[packet.seq_num] == _flow_id) {
      const unsigned int lost_since_last_time = (unsigned int) max(packet.seq_num-_largest_ack-1, (long) 0);
      _packets_lost += lost_since_last_time;
      _memory.lost(lost_since_last_time);
    }

    for (auto it=_id_to_sent_during_action.begin(); it!=_id_to_sent_during_action.lower_bound(packet.seq_num); it++) {
      _id_to_sent_during_action.erase(it->first);
      for (auto inner_it=_flow_to_last_received.begin(); inner_it!=_flow_to_last_received.lower_bound(_id_to_sent_during_flow[it->first]);) {
        inner_it = _flow_to_last_received.erase(inner_it);
      }
      _id_to_sent_during_flow.erase(it->first);
    }

    int packets_sent_in_this_episode = 1;
    // if (!packet.first) {
      packets_sent_in_this_episode = (int) _outstanding_rewards[_id_to_sent_during_action[packet.seq_num]]["sent"];

      const double delay = packet.tick_received - packet.tick_sent;

      for (auto it=_outstanding_rewards.begin(); it!=_outstanding_rewards.lower_bound(_id_to_sent_during_action[packet.seq_num]);) {
        const double throughput_final = it->second["received"];
        const double delay_final = it->second["delay_acc"];
        // const double duration = it->second["end_time"] - it->second["start_time"];
        // const double duration = it->second["intersend_duration_acc"];
        const double duration = it->second["interreceive_duration_acc"];
        const double lost_final = it->second["sent"] - it->second["received"];
        if (_training) {
          // FIXME: A debugging hack to assert that no packets get lost when using infinitely sized buffers
          // assert(it->second["received"] == it->second["sent"]);
          _rainbow.put_reward(_thread_id, throughput_final, delay_final, duration, lost_final);
        }
        _put_rewards += 1;
        it = _outstanding_rewards.erase(it);
      }

      _outstanding_rewards[_id_to_sent_during_action[packet.seq_num]]["received"] += 1;
      _outstanding_rewards[_id_to_sent_during_action[packet.seq_num]]["delay_acc"] += delay;
      // FIXME: It doesn't really matter but conceptually it's wrong. It should be the time since the start of the simulation, for example
      // if (_memory._last_tick_received != 0) {
        _outstanding_rewards[_id_to_sent_during_action[packet.seq_num]]["interreceive_duration_acc"] += packet.tick_received - _flow_to_last_received[_id_to_sent_during_flow[packet.seq_num]];
        assert(_outstanding_rewards[_id_to_sent_during_action[packet.seq_num]]["interreceive_duration_acc"] > 0);
      // } else {
      //   _outstanding_rewards[_id_to_sent_during_action[packet.seq_num]]["interreceive_duration_acc"] += packet.tick_received - _start_tick;
      // }
      // _outstanding_rewards[_id_to_sent_during_action[packet.seq_num]]["end_time"] = packet.tick_received;
    // }

    _packets_received += 1;

    // printf("%lu: %u, %u\n", _thread_id, _id_to_sent_during_flow[packet.seq_num], _flow_id);
    if (_id_to_sent_during_flow[packet.seq_num] == _flow_id && (_active_flows.find(_id_to_sent_during_flow[packet.seq_num]) != _active_flows.end())) {
      // printf("%lu: Got packet flow of packet=%u, flow of Unicorn=%u, sent=%f, received=%f, seq_num=%d\n", _thread_id, _id_to_sent_during_flow[packet.seq_num], _flow_id, packet.tick_sent, packet.tick_received, packet.seq_num);
      vector<remy::Packet> packet_for_memory_update;
      packet_for_memory_update.push_back(packet);
      _memory.packets_received(packet_for_memory_update, _flow_id, _largest_ack );

      // printf("%lu: Yeah, getting action after receiving a packet...\n", _thread_id);
      get_action(packet.tick_received, packets_sent_in_this_episode);
    }

    _largest_ack = max(packet.seq_num, _largest_ack);

    _flow_to_last_received[_id_to_sent_during_flow[packet.seq_num]] = packet.tick_received;

    // for (auto it=_flow_to_last_received.begin(); it!=_flow_to_last_received.lower_bound(_id_to_sent_during_flow[packet.seq_num]);) {
    //   it = _flow_to_last_received.erase(it);
    // }

    // _id_to_sent_during_flow.erase(packet.seq_num);
    // _id_to_sent_during_action.erase(packet.seq_num);

    // // FIXME: This would be nicer
    // for (auto it=_flow_to_last_received.begin(); it!=_flow_to_last_received.lower_bound(_id_to_sent_during_flow[packet.seq_num]);) {
    //   it = _flow_to_last_received.erase(it);
    // }
    // for (auto it=_id_to_sent_during_flow.begin(); it!=_id_to_sent_during_flow.lower_bound(packet.seq_num);) {
    //   it = _id_to_sent_during_flow.erase(it);
    // }
    // for (auto it=_id_to_sent_during_action.begin(); it!=_id_to_sent_during_action.lower_bound(packet.seq_num);) {
    //   it = _id_to_sent_during_action.erase(it);
    // }
  }

  // _largest_ack = max( packets.at( packets.size() - 1 ).seq_num, _largest_ack );
  // if (!(_largest_ack > previous_largest_ack)) {
    // printf("%lu: largest ack: %d, previous largest ack: %d\n", _thread_id, _largest_ack, previous_largest_ack);
  // }
  // assert (_largest_ack > previous_largest_ack);
}

void Unicorn::reset(const double & tickno)
{
  // printf("%lu: Fucking resetting\n", _thread_id);
  _rainbow._training = _training;
  // assert(false);
  // printf("%lu: Resetting\n", _thread_id);
  // _largest_ack -= 1;
  if (_outstanding_rewards.size() != _put_actions-_put_rewards) {
    printf("%lu: _outstanding_rewards=%lu, _put_actions=%lu, _put_rewards=%lu\n", _thread_id, _outstanding_rewards.size(), _put_actions, _put_rewards);
  }
  assert(_outstanding_rewards.size() == _put_actions-_put_rewards);
  if (_thread_id > 0) {
    // printf("%lu: Lost rewards at reset\n", _thread_id);
    // put_lost_rewards(_packets_sent-_largest_ack);
    // _rainbow.put_reward(_thread_id, LOSS_REWARD);
    // _put_rewards += 1;
    finishFlow();
  }

  // FIXME: Then why exactly do we also do this in unicorn-template.cc?
  if (_active_flows.find(_flow_id) != _active_flows.end()) {
    _active_flows.erase(_flow_id);
  }

  // if (_put_actions != _put_rewards) {
  //   printf("%lu: _put_actions: %lu, _put_rewards: %lu\n", _thread_id, _put_actions, _put_rewards);
  // }
  // Cannot assert that anymore since now we actually always process all rewards.
  // assert(_put_actions == _put_rewards);
  // assert(_sent_packets.size() == 0);

  // FIXME: Is this good?
  // _memory.reset();
  _memory._lost_since_last_time = 0;

  _largest_ack = _packets_sent - 1; /* Assume everything's been delivered */
  _last_send_time = 0;
  _the_window = MIN_WINDOW_UNICORN; // Reset the window to 1
  _flow_id += 1;
  _active_flows.insert(_flow_id);
  // _largest_ack = _packets_sent - 1; /* Assume everything's been delivered */
  // _put_actions = 0;
  // _put_rewards = 0;
  // _outstanding_rewards.clear();
  // _id_to_sent_during_action.clear();
  _start_tick = tickno;

  assert( _flow_id != 0 );

  if (_thread_id == 0) {
    _thread_id = _rainbow.create_thread(_delay_delta);
    // printf("Assigned thread id %lu to Unicorn\n", _thread_id);
  }
  // printf("%lu: Starting\n", _thread_id);
  get_action(tickno);

  /* initial window and intersend time */
  // const Whisker & current_whisker( _whiskers.use_whisker( _memory, _track ) );
  // _the_window = current_whisker.window( _the_window );
  // _intersend_time = current_whisker.intersend();
}

double Unicorn::next_event_time( const double & tickno )
{
  // return tickno;
  if (int(_packets_sent) < _largest_ack + 1 + (int) floor(_the_window) ) {
    return tickno;
  } else if (_last_send_time > 0 && tickno - _last_send_time >= TIMEOUT_THRESHOLD) {
    reset(tickno);
    return tickno;
  } else {
    /* window is currently closed */
    return std::numeric_limits<double>::max();
  }
}

const double NORMALIZER = 1e-2;

void Unicorn::get_action(const double& tickno, const int& packets_sent_in_this_episode) {

  // FIXME: HACK!!!
  // (void) packets_sent_in_this_episode;
  const double action = _rainbow.get_action(
    _thread_id,
    {
      _memory.field(0)*NORMALIZER,
      _memory.field(1)*NORMALIZER,
      _memory.field(2),
      _memory.field(3)*NORMALIZER,
      _memory.field(4)*NORMALIZER,
      _memory.field(6)*NORMALIZER,
      (_memory._last_tick_received - _memory._last_tick_sent)*NORMALIZER,
      (double) _the_window,
      // _memory.field(6), // loss rate
      // (double) tickno - _memory._last_tick_sent, // time since last send
      // (double) tickno - _memory._last_tick_received, // time since last receive
      // (double) _memory._lost_since_last_time, // losses since last receive
      (double) packets_sent_in_this_episode,
      _memory._send*NORMALIZER,
      _memory._rec*NORMALIZER,
      (double) _memory._lost_since_last_time,
      // _memory.field(2),
      // _memory.field(4),
      // (double) _the_window,
      // (double) tickno - _last_send_time
    },
    tickno,
    _the_window
  );
  // action.intersend /= 100.0;
  // printf("%lu: action is: %f, %f, %f\n", _thread_id, action.window_increment, action.window_multiple, action.intersend);
  // printf("%lu: action is: %f\n", _thread_id, action);
  _put_actions += 1;

  // printf("%lu: window=%f, action=%f\n", _thread_id, _the_window, action);
  // _the_window = window(_the_window, action.window_increment, action.window_multiple);
  _the_window = std::min(std::max(_the_window + action, MIN_WINDOW_UNICORN), MAX_WINDOW_UNICORN);
  // if (!_training) printf("%lu: window: %d\n", _thread_id, _the_window);
  // printf("%lu: after update: window=%f, action=%f\n", _thread_id, _the_window, action);
  for (auto it=_outstanding_rewards.begin(); it!=_outstanding_rewards.end(); it++) {
    if (it->second["end_time"] < 0) {
      it->second["end_time"] = tickno;
    }
  }
  _outstanding_rewards[_put_actions] = {{"interreceive_duration_acc", 0.0}, {"intersend_duration_acc", 0.0}, {"sent", 0.0}, {"received", 0.0}, {"start_time", tickno}, {"end_time", -1.0}, {"delay_acc", 0.0}};
  // _outstanding_rewards.push_back({int(floor(_the_window)), 0, 0, tickno, -1.0, 0.0, _packets_sent + int(floor(_the_window))});
}

void Unicorn::finishFlow() {
  // const bool at_least_one_packet_sent = _put_actions>1;
  // if (!at_least_one_packet_sent) {
  //   _rainbow.put_reward(_thread_id, LOSS_REWARD);
  //   _put_rewards += 1;
  // }
  // put_lost_rewards();
  // const bool at_least_one_packet_sent = true;
  // printf("%lu: finish, _packets_sent: %u\n", _thread_id, _packets_sent);
  // _rainbow.finish(_thread_id, {_memory.field(0), _memory.field(1), _memory.field(2), _memory.field(3), _memory.field(6), (double)_the_window/WINDOW_NORMALIZER}, at_least_one_packet_sent);
  // printf("%lu: Finishing, window=%f\n", _thread_id, _the_window);
  _rainbow.finish(_thread_id, _outstanding_rewards.size(), _memory._last_tick_sent-_start_tick, _the_window);
  // printf("%lu: actions: %lu, rewards: %lu, outstanding_rewards: %lu\n", _thread_id, _put_actions, _put_rewards, _outstanding_rewards.size());
  // _put_actions -= _outstanding_rewards.size();
}

// void Unicorn::put_lost_rewards() {
  // printf("%lu: Going to put loss rewards for %d intervals\n", _thread_id, number);
  // for (auto &tuple : _outstanding_rewards) {
  //   get<2>(tuple) = get<0>(tuple) - tuple["received"];
  //   const double throughput_final = ALPHA*log(tuple["received"]/(tuple["end_time"]-tuple["start_time"]));
  //   const double delay_final = BETA*log(tuple["delay_acc"]/tuple["received"]);
  //   printf("%lu: end time: %f, start time: %f, lost: %u, received: %u, window: %u\n", _thread_id, tuple["end_time"], tuple["start_time"], get<2>(tuple), tuple["received"], get<0>(tuple));
  //   printf("%lu: Calculated reward at lost packets delay:%f, throughput:%f\n", _thread_id, -delay_final, throughput_final);
  //   _rainbow.put_reward(_thread_id, throughput_final-delay_final);
  //   _put_rewards += 1;
  // }
// }

Unicorn::~Unicorn() {
  // printf("Destroying Unicorn with thread id %lu\n", _thread_id);
  if (_thread_id > 0) {
    // printf("%lu: Lost rewards at destruction\n", _thread_id);
    // put_lost_rewards(_packets_sent-_largest_ack);
    // _rainbow.put_reward(_thread_id, LOSS_REWARD);
    // _put_rewards += 1;
    finishFlow();
    _rainbow.delete_thread(_thread_id);
  }
}

SimulationResultBuffers::SenderState Unicorn::state_DNA() const
{
  SimulationResultBuffers::SenderState ret;
  ret.mutable_memory()->CopyFrom( _memory.DNA() );
  ret.set_window_size( _the_window );
  ret.set_intersend_time( nan("1") );
  return ret;
}
