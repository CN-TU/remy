# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import numpy.random
# import scipy as sp
# from scipy.stats import norm, gamma
import random
import time
import sys
import math

from game_ac_network import GameACLSTMNetwork, tiny#, GameACFFNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import USE_LSTM
# from constants import ACTION_SIZE
from constants import ALPHA, BETA
from constants import LOG_LEVEL

import logging
logging.basicConfig(level=LOG_LEVEL)

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device):

    self.states = []
    self.actions = []
    self.rewards = []
    self.durations = []
    self.values = []

    self.start_lstm_states = []

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    if USE_LSTM:
      self.local_network = GameACLSTMNetwork(thread_index, device)
    else:
      self.local_network = GameACFFNetwork(thread_index, device)

    self.local_network.prepare_loss(ENTROPY_BETA)

    with tf.device(device):
      var_refs = [v._ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(
        self.local_network.total_loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.gradients )
      
    self.sync = self.local_network.sync_from(global_network)
        
    self.local_t = 0
    self.episode_start_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward_throughput = 0
    self.episode_reward_delay = 0

    # variable controlling log output
    # self.prev_local_t = 0

    # self.acc_state = None

  def get_network_vars(self):
    return self.local_network.get_vars()

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def _record_score(self, sess, summary_writer, summary_op, summary_inputs, things, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      summary_inputs["score_throughput"]: things["score_throughput"],
      summary_inputs["score_delay"]: things["score_delay"],
      summary_inputs["entropy"]: things["entropy"],
      summary_inputs["action_loss"]: things["action_loss"],
      summary_inputs["value_loss"]: things["value_loss"],
      summary_inputs["total_loss"]: things["total_loss"],
      summary_inputs["window"]: things["window"],
      summary_inputs["std"]: things["std"]
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  def action_step(self, sess, state):

    if len(self.actions) % LOCAL_T_MAX == 0:
      # Sync for the next iteration
      sess.run( self.sync )
      if USE_LSTM:
        self.start_lstm_states.append(self.local_network.lstm_state_out)

    pi_, action, value_ = self.local_network.run_policy_action_and_value(sess, state)

    logging.debug(" ".join(map(str,(self.thread_index,"pi_values:",pi_))))

    self.states.append(state)
    self.actions.append(action)

    action = action[0]

    self.values.append(value_)
    if self.local_t % LOG_INTERVAL == 0:
      logging.debug("{}: pi={}".format(self.thread_index, pi_))
      logging.debug("{}: V={}".format(self.thread_index, value_))
    return action

  def reward_step(self, sess, global_t, summary_writer, summary_op, summary_inputs, reward_throughput, reward_delay, duration):
    assert(reward_throughput >= 0)
    assert(reward_delay >= 0)
    self.rewards.append((reward_throughput, reward_delay))
    assert(duration >= 0)
    self.durations.append(duration)

    if len(self.rewards)>LOCAL_T_MAX:
      return self.process(sess, global_t, summary_writer, summary_op, summary_inputs)
    else:
      return  0
    # implicitly returns None otherwise

  def final_step(self, sess, global_t, summary_writer, summary_op, summary_inputs, actions_to_remove, time_difference):
    self.actions = self.actions[:-actions_to_remove]
    self.states = self.states[:-actions_to_remove]
    self.values = self.values[:-actions_to_remove]

    if len(self.rewards) > 0:
      return self.process(sess, global_t, summary_writer, summary_op, summary_inputs, time_difference)
    else:
      return 0

  def process(self, sess, global_t, summary_writer, summary_op, summary_inputs, time_difference=None):

    final = time_difference is not None
    start_local_t = self.local_t
    
    logging.debug(" ".join(map(str,(self.thread_index, "In process: len(rewards)", len(self.rewards), "len(durations)", len(self.durations), "len(states)", len(self.states), "len(actions)", len(self.actions), "len(values)", len(self.values)))))

    actions = self.actions[:LOCAL_T_MAX]
    states = self.states[:LOCAL_T_MAX]
    rewards = self.rewards[:LOCAL_T_MAX]
    durations = self.durations[:LOCAL_T_MAX]
    values = self.values[:LOCAL_T_MAX]

    logging.debug(" ".join(map(str,(self.thread_index, "In process: rewards", rewards, "durations", durations, "states", states, "actions", actions, "values", values))))

    # t_max times loop
    # for i in range(len(rewards)):
    #   self.local_t += 1

    self.local_t += len(rewards)

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t

    if final:
      # R_throughput = R_delay = 0.0
      # FIXME: This is a hack! It should evaluate how good the very last state is after the last action
      R_packets, R_accumulated_delay, R_duration = self.local_network.run_value(sess, self.states[-1])
      log_str = "final"
    else:
      R_packets, R_accumulated_delay, R_duration = self.local_network.run_value(sess, self.states[LOCAL_T_MAX])
      log_str = "intermediate"
    logging.debug(log_str+": "+" ".join(map(str,("R_packets", R_packets, "R_accumulated_delay", R_accumulated_delay, "R_duration", R_duration))))
    # R_throughput = R_delay = 0.0
    # try:
    #   R_packets = R_throughput*R_duration
    #   R_accumulated_delay = R_delay*R_packets
    # except Exception as e:
    #   print("An exception occurred while multiplying the values:", e)
    #   raise e

    # R_packets, R_accumulated_delay, R_duration = np.exp(R_packets), np.exp(R_accumulated_delay), np.exp(R_duration)
    # logging.debug(" ".join(map(str,("exp(R_packets)", R_packets, "exp(R_accumulated_delay)", R_accumulated_delay, "exp(R_duration)", R_duration))))
    assert(np.isfinite(R_duration))
    assert(np.isfinite(R_packets))
    assert(np.isfinite(R_accumulated_delay))

    actions.reverse()
    states.reverse()
    rewards.reverse()
    durations.reverse()
    values.reverse()
    # logging.debug(" ".join(map(str,("values", values))))

    batch_si = []
    batch_ai = []
    batch_td_throughput = []
    batch_td_delay = []
    batch_R_duration = []
    batch_R_packets = []
    batch_R_accumulated_delay = []

    # compute and accmulate gradients
    for(ai, ri, di, si, Vi) in zip(actions, rewards, durations, states, values):

      R_duration = di + GAMMA * R_duration

      R_packets = ri[0] + GAMMA * R_packets
      # R_throughput = R_packets/R_duration
      # FIXME: Put assertions here
      td_throughput = np.log(R_packets/R_duration) - np.log(Vi[0]/Vi[2])

      R_accumulated_delay = ri[1] + GAMMA * R_accumulated_delay
      # R_delay = R_accumulated_delay/R_packets
      td_delay = -(np.log(R_accumulated_delay/R_packets) - np.log(Vi[1]/Vi[0]))

      batch_si.append(si)
      batch_ai.append(ai)
      batch_td_throughput.append(ALPHA*td_throughput)
      batch_td_delay.append(BETA*td_delay)
      batch_R_duration.append(R_duration)
      # batch_R_duration.append(np.log(R_duration))
      batch_R_packets.append(R_packets)
      # batch_R_packets.append(np.log(R_packets))
      batch_R_accumulated_delay.append(R_accumulated_delay)
      # batch_R_accumulated_delay.append(np.log(R_accumulated_delay))

      logging.debug(" ".join(map(str,("batch_td_throughput[-1]", batch_td_throughput[-1], "batch_td_delay[-1]", batch_td_delay[-1], "batch_R_packets[-1]", batch_R_packets[-1], "batch_R_accumulated_delay[-1]", batch_R_accumulated_delay[-1], "batch_R_duration[-1]", batch_R_duration[-1]))))

      self.episode_reward_throughput += ri[0]
      self.episode_reward_delay += ri[1]

    cur_learning_rate = self._anneal_learning_rate(global_t)

    # logging.debug(" ".join(map(str,("All the batch stuff", "batch_si", batch_si, "batch_ai", batch_ai, "batch_td_throughput", batch_td_throughput, "batch_td_delay", batch_td_delay,"batch_R_packets", batch_R_packets, "batch_R_accumulated_delay", batch_R_accumulated_delay, "batch_R_duration", batch_R_duration))))

    if USE_LSTM:
      batch_si.reverse()
      batch_ai.reverse()
      batch_td_throughput.reverse()
      batch_td_delay.reverse()
      batch_R_duration.reverse()
      batch_R_packets.reverse()
      batch_R_accumulated_delay.reverse()

      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_ai,
                  self.local_network.td_throughput: batch_td_throughput,
                  self.local_network.td_delay: batch_td_delay,
                  self.local_network.r_duration: batch_R_duration,
                  self.local_network.r_packets: batch_R_packets,
                  self.local_network.r_accumulated_delay: batch_R_accumulated_delay,
                  self.local_network.initial_lstm_state: self.start_lstm_states[0],
                  self.local_network.step_size : [len(batch_ai)],
                  self.learning_rate_input: cur_learning_rate } )
    else:
      raise NotImplementedError("FF currently not implemented.")

    if final:
      normalized_final_score_throughput = self.episode_reward_throughput/time_difference
      normalized_final_score_delay = self.episode_reward_delay/self.episode_reward_throughput
      logging.debug("{}: score_throughput={}, score_delay={}".format(self.thread_index, normalized_final_score_throughput, normalized_final_score_delay))
      entropy, action_loss, value_loss, total_loss, window, std = self.local_network.run_loss(sess, batch_si[-1], batch_ai[-1], batch_td_throughput[-1], batch_td_delay[-1], batch_R_duration[-1], batch_R_packets[-1], batch_R_accumulated_delay[-1])
      things = {"score_throughput": normalized_final_score_throughput, 
        "score_delay": normalized_final_score_delay, 
        "action_loss": action_loss.item(),
        "value_loss": value_loss,
        "entropy": entropy.item(),
        "total_loss": total_loss,
        "window": window.item(),
        "std": std.item()}
      logging.debug(" ".join(map(str,("things", things))))
      self._record_score(sess, summary_writer, summary_op, summary_inputs, things, global_t) # TODO:NOW: is that "not terminal_end" correct?
      self.episode_start_t = self.local_t
      self.episode_reward_throughput = 0
      self.episode_reward_delay = 0

      elapsed_time = time.time() - self.start_time
      steps_per_sec = self.local_t / elapsed_time
      logging.info("### {}: Performance: {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(self.thread_index, self.local_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

      self.local_t = 0

      if USE_LSTM:
        self.local_network.reset_state()

    self.actions = self.actions[LOCAL_T_MAX:]
    self.states = self.states[LOCAL_T_MAX:]
    self.values = self.values[LOCAL_T_MAX:]
    self.rewards = self.rewards[LOCAL_T_MAX:]
    self.durations = self.durations[LOCAL_T_MAX:]
    self.start_lstm_states = self.start_lstm_states[1:]

    return diff_local_t
    
