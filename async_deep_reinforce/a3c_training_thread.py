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

from game_ac_network import GameACLSTMNetwork

from constants import GAMMA
# gamma_current, gamma_future = 1./(1+GAMMA), GAMMA/(1.+GAMMA)
from constants import LOCAL_T_MAX
from constants import LOG_LEVEL
from constants import SECONDS_NORMALIZER

import logging

logging.basicConfig(level=LOG_LEVEL)

LOG_INTERVAL = 100

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device,
               training,
               cooperative,
               delay_delta):

    self.delay_delta = delay_delta
    logging.info(" ".join(map(str,("delay_delta", delay_delta))))
    self.training = training
    self.cooperative = cooperative
    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    self.local_network = GameACLSTMNetwork(thread_index, device)
    self.local_network.prepare_loss()

    with tf.device(device):
      var_refs = [v._ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(
        self.local_network.total_loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

      self.apply_gradients = grad_applier.apply_gradients(
        zip(self.gradients, global_network.get_vars())
      )

    self.sync = self.local_network.sync_from(global_network)
    self.episode_count = 0

    self.backup_vars = self.local_network.backup_vars()
    self.restore_backup = self.local_network.restore_backup()

    self.initial_learning_rate = initial_learning_rate

  def get_network_vars(self):
    return self.local_network.get_vars()

  def _anneal_learning_rate(self, global_time_step):
    # learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    # if learning_rate < 0.0:
    #   learning_rate = 0.0
    # return learning_rate
    return self.initial_learning_rate

  def _record_score(self, sess, summary_writer, summary_op, summary_inputs, things, global_t):
    # print("window in _record_score", self.windows, self.time_differences)
    feed_dict = {}
    for key in things.keys():
      feed_dict[summary_inputs[key]] = things[key]
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

  def action_step(self, sess, state, tickno, window):
    # Run this still with the old weights, before syncing them
    # print("state", state)

    # print(self.thread_index, "state", state)
    if self.training:
      self.estimated_values.append(self.local_network.run_value(sess, state))

      if len(self.actions) % LOCAL_T_MAX == 0:
        self.time_differences.append(None)
        # Sync for the next iteration
        sess.run( self.sync )
        self.start_lstm_states.append((self.local_network.lstm_state_out_action, self.local_network.lstm_state_out_value))
        self.variable_snapshots.append(sess.run(self.local_network.get_vars()))

    if self.training:
      action, value_ = self.local_network.run_action_and_value(sess, state)
    else:
      action = self.local_network.run_action(sess, state)

    # logging.debug(" ".join(map(str,(self.thread_index,"pi_values:",pi_))))

    if self.training:
      self.states.append(state)
      self.ticknos.append(tickno)
      self.windows.append(window)
      self.actions.append(action)
      self.values.append(value_)
    # if self.local_t % LOG_INTERVAL == 0:
    #   logging.debug("{}: pi={}".format(self.thread_index, pi_))
    #   logging.debug("{}: V={}".format(self.thread_index, value_))
    return action[0]

  def reward_step(self, sess, global_t, summary_writer, summary_op, summary_inputs, reward_throughput, reward_delay, duration):
    assert(reward_throughput >= 0)
    assert(reward_delay >= 0)
    self.rewards.append((reward_throughput, reward_delay))
    assert(duration >= 0)
    self.durations.append(duration)

    if len(self.rewards)>=LOCAL_T_MAX or (len([item for item in self.actions[:LOCAL_T_MAX] if item is not None]) == len(self.rewards) and len(self.rewards) > 0 and self.time_differences[0] is not None):
      # print(self.thread_index, "rewards", self.rewards, "actions", self.actions, "time_diffs", self.time_differences)
      assert(len(self.rewards) <= LOCAL_T_MAX)
      # print(len([item for item in self.actions[:LOCAL_T_MAX] if item is not None]), len(self.rewards[:LOCAL_T_MAX]))
      assert(len([item for item in self.actions[:LOCAL_T_MAX] if item is not None]) == len(self.rewards[:LOCAL_T_MAX]))
      result = self.process(sess, global_t, summary_writer, summary_op, summary_inputs, self.time_differences[0])
      return result
    else:
      return  0

  def final_step(self, sess, global_t, summary_writer, summary_op, summary_inputs, actions_to_remove, time_difference, window):
    # print(self.thread_index, "self.time_differences", self.time_differences)
    # print("self.actions", len(self.actions))
    # print("self.states", len(self.states))
    # print("self.values", len(self.values))
    # print("self.rewards", len(self.rewards))
    # print("self.durations", len(self.durations))
    # print("self.estimated_values", len(self.estimated_values))
    # print("self.time_differences", len(self.time_differences))
    # print("self.start_lstm_states", len(self.start_lstm_states))
    # print("self.variable_snapshots", len(self.variable_snapshots))
    # self.actions = self.actions[:-actions_to_remove]
    # self.states = self.states[:-actions_to_remove]
    # self.values = self.values[:-actions_to_remove]
    # self.estimated_values = self.estimated_values[:-actions_to_remove+1]
    # print("Final step is called")

    if self.training:
      if len(self.actions) > 0:
        self.time_differences = self.time_differences[:-1]
        self.time_differences.append(time_difference)
        # self.windows = self.windows[:-1]
        # self.windows.append(window)

        nones_to_add = [None] * ((LOCAL_T_MAX - (len(self.actions) % LOCAL_T_MAX)) % LOCAL_T_MAX)
        self.actions += nones_to_add
        self.states += nones_to_add
        self.values += nones_to_add
        self.estimated_values += nones_to_add
        self.windows += nones_to_add
        self.ticknos += nones_to_add
      # TODO: Is this useful? I guess only the `local_t' is actually needed...
      elif len(self.actions) <= 0:
        self.episode_count += 1
        self.local_t = 0
        self.episode_reward_throughput = 0
        self.episode_reward_delay = 0

    # If, for some strange reason, absolutely nothing happened in this episode, don't do anything...
    # Or if you're actually in testing mode :)
    # if len(self.rewards)>0:
    #   time_diff = self.process(sess, global_t, summary_writer, summary_op, summary_inputs, time_difference)
    # else:
    #   time_diff = 0

    # self.states = []
    # self.actions = []
    # self.rewards = []
    # self.durations = []
    # self.values = []
    # self.estimated_values = []
    # self.start_lstm_states = []
    # self.variable_snapshots = []
    self.local_network.reset_state()

  def process(self, sess, global_t, summary_writer, summary_op, summary_inputs, time_difference=None):
    if self.local_t <= 0:
      self.start_time = time.time()

    final = time_difference is not None

    start_local_t = self.local_t

    # logging.debug(" ".join(map(str,(self.thread_index, "In process: len(rewards)", len(self.rewards), "len(durations)", len(self.durations), "len(states)", len(self.states), "len(actions)", len(self.actions), "len(values)", len(self.values)))))

    actions = [item for item in self.actions[:LOCAL_T_MAX] if item is not None]
    ticknos = [item for item in self.ticknos[:LOCAL_T_MAX] if item is not None]
    windows = [item for item in self.windows[:LOCAL_T_MAX] if item is not None]
    states = [item for item in self.states[:LOCAL_T_MAX] if item is not None]
    rewards = [item for item in self.rewards[:LOCAL_T_MAX] if item is not None]
    durations = [item for item in self.durations[:LOCAL_T_MAX] if item is not None]
    values = [item for item in self.values[:LOCAL_T_MAX] if item is not None]

    assert(len(actions) > 0)
    assert(len(ticknos) > 0)
    assert(len(windows) > 0)
    assert(len(states) > 0)
    assert(len(rewards) > 0)
    assert(len(durations) > 0)
    assert(len(values) > 0)
    if not (len(actions) == len(ticknos) == len(windows) == len(states) == len(rewards) == len(durations) == len(values)):
      print(len(self.actions), len(self.ticknos), len(self.windows), len(self.states), len(self.rewards), len(self.durations), len(self.values))
      print(len(actions), len(ticknos), len(windows), len(states), len(rewards), len(durations), len(values))
      print(self.actions, self.ticknos, self.windows, self.states, self.rewards, self.durations, self.values)
      print(actions, ticknos, windows, states, rewards, durations, values)
    assert(len(actions) == len(ticknos) == len(windows) == len(states) == len(rewards) == len(durations) == len(values))

    assert(len(self.actions) == len(self.ticknos) == len(self.windows) == len(self.states) == len(self.values) == len(self.estimated_values))
    assert(len(self.rewards) == len(self.durations))
    assert(len(self.time_differences) == len(self.start_lstm_states) == len(self.variable_snapshots))

    # logging.debug(" ".join(map(str,(self.thread_index, "In process: rewards", rewards, "durations", durations, "states", states, "actions", actions, "values", values))))

    # get estimated value of step n+1
    # assert((not len(self.estimated_values) <= len(rewards)) or final)
    # print("self.estimated_values", self.estimated_values)
    # print("Spam and eggs")
    R_packets, R_accumulated_delay, R_duration = self.estimated_values[len(rewards)] if self.estimated_values[len(rewards)] is not None else self.estimated_values[len(rewards)-1]

    R_packets_initial, R_accumulated_delay_initial, R_duration_initial = R_packets, R_accumulated_delay, R_duration

    R_packets, R_accumulated_delay, R_duration = (R_packets)/(1-GAMMA), (R_accumulated_delay)/(1-GAMMA), (R_duration)/(1-GAMMA)
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
    batch_td = []
    batch_R_duration = []
    batch_R_packets = []
    batch_R_accumulated_delay = []

    # compute and accmulate gradients
    for(ai, ri, di, si, Vi) in zip(actions, rewards, durations, states, values):

      R_duration = (di*SECONDS_NORMALIZER + GAMMA*R_duration)

      R_packets = (ri[0] + GAMMA*R_packets)
      # TODO: In theory it should be np.log(R_bytes/R_duration) but remy doesn't have bytes
      td = (np.log(R_packets/R_duration) - np.log(Vi[0]/Vi[2]))

      R_accumulated_delay = (ri[1]*SECONDS_NORMALIZER + GAMMA*R_accumulated_delay)
      # R_delay = R_accumulated_delay/R_packets
      # td_delay = -(np.log(R_accumulated_delay/R_packets/DELAY_MULTIPLIER) - np.log(Vi[1]/Vi[0]/DELAY_MULTIPLIER))
      td -= self.delay_delta/SECONDS_NORMALIZER*(R_accumulated_delay/R_packets - Vi[1]/Vi[0])
      # td_delay = -(R_accumulated_delay/R_packets - Vi[1]/Vi[0])

      batch_si.append(si)
      batch_ai.append(ai)
      batch_td.append(td)
      batch_R_duration.append(R_duration*(1-GAMMA))
      # batch_R_duration.append(np.log(R_duration))
      batch_R_packets.append(R_packets*(1-GAMMA))
      # batch_R_packets.append(np.log(R_packets))
      batch_R_accumulated_delay.append(R_accumulated_delay*(1-GAMMA))
      # batch_R_accumulated_delay.append(np.log(R_accumulated_delay))

      # logging.debug(" ".join(map(str,("batch_td_throughput[-1]", batch_td_throughput[-1], "batch_td_delay[-1]", batch_td_delay[-1], "batch_R_packets[-1]", batch_R_packets[-1], "batch_R_accumulated_delay[-1]", batch_R_accumulated_delay[-1], "batch_R_duration[-1]", batch_R_duration[-1]))))

      self.episode_reward_throughput += ri[0]
      self.episode_reward_delay += ri[1]

    self.local_t += len(rewards)

    # if final or self.local_t % LOG_INTERVAL == 0:
    #   print(self.thread_index, "actions", len(self.actions), "rewards", len(self.rewards), "values", len(self.values), "estimated_values", len(self.estimated_values))
    #   print(self.thread_index, "actions", self.actions, "rewards", self.rewards, "values", self.values, "estimated_values", self.estimated_values)

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t

    cur_learning_rate = self._anneal_learning_rate(global_t)

    # logging.debug(" ".join(map(str,("All the batch stuff", "batch_si", batch_si, "batch_ai", batch_ai,"batch_R_packets", batch_R_packets, "batch_R_accumulated_delay", batch_R_accumulated_delay, "batch_R_duration", batch_R_duration))))

    self.backup_vars()

    batch_si.reverse()
    batch_ai.reverse()
    batch_td.reverse()
    batch_R_duration.reverse()
    batch_R_packets.reverse()
    batch_R_accumulated_delay.reverse()

    feed_dict = {
      self.local_network.s: batch_si,
      self.local_network.a: batch_ai,
      self.local_network.td: batch_td,
      self.local_network.r_duration: batch_R_duration,
      self.local_network.r_packets: batch_R_packets,
      self.local_network.r_accumulated_delay: batch_R_accumulated_delay,
      self.local_network.initial_lstm_state_action: self.start_lstm_states[0][0],
      self.local_network.initial_lstm_state_value: self.start_lstm_states[0][1],
      self.local_network.step_size : [len(batch_ai)],
      self.learning_rate_input: cur_learning_rate
    }
    var_dict = dict(zip(self.local_network.get_vars(), self.variable_snapshots[0]))
    feed_dict.update(var_dict)

    sess.run( self.apply_gradients,
              feed_dict = feed_dict )

    # if len(ticknos) == 0:
    #   print(self.thread_index, "actions", self.actions, "rewards", self.rewards, "values", self.values, "estimated_values", self.estimated_values, "ticknos", self.ticknos)
    # if final or self.local_t % LOG_INTERVAL == 0:
    if final:
      if ticknos[-1]-ticknos[0] > 0 and self.episode_reward_throughput > 0:
        # print(ticknos)
        # print(self.episode_reward_throughput, ticknos[0], ticknos[-1])
        # normalized_final_score_throughput = self.episode_reward_throughput/(ticknos[-1]-ticknos[0])
        # logging.info("{}: self.episode_reward_throughput={}, time_difference={}".format(self.thread_index, self.episode_reward_throughput, time_difference))
        normalized_final_score_delay = self.episode_reward_delay/self.episode_reward_throughput
        # print(self.windows)

        logging.info("final:"+str(final)+", delta:"+str(self.delay_delta)+"; "+" ".join(map(str,("R_packets", R_packets_initial, "R_accumulated_delay", R_accumulated_delay_initial/SECONDS_NORMALIZER, "R_duration", R_duration_initial/SECONDS_NORMALIZER))))
        # logging.info("{}: score_throughput={}, score_delay={}, measured throughput beginning={}, measured delay beginning={}, measured throughput end={}, measured delay end={}".format(self.thread_index, normalized_final_score_throughput, normalized_final_score_delay, batch_R_packets[0]/batch_R_duration[0]*SECONDS_NORMALIZER, batch_R_accumulated_delay[0]/batch_R_packets[0]/SECONDS_NORMALIZER, batch_R_packets[-1]/batch_R_duration[-1]*SECONDS_NORMALIZER, batch_R_accumulated_delay[-1]/batch_R_packets[-1]/SECONDS_NORMALIZER))
        logging.info("{}: score_delay={}, measured throughput beginning={}, measured delay beginning={}, measured throughput end={}, measured delay end={}".format(self.thread_index, normalized_final_score_delay, batch_R_packets[0]/batch_R_duration[0]*SECONDS_NORMALIZER, batch_R_accumulated_delay[0]/batch_R_packets[0]/SECONDS_NORMALIZER, batch_R_packets[-1]/batch_R_duration[-1]*SECONDS_NORMALIZER, batch_R_accumulated_delay[-1]/batch_R_packets[-1]/SECONDS_NORMALIZER))

        # time_difference > 0 because of a bug in Unicorn.cc that makes it possible for time_difference to be smaller than 0.

        # elapsed_time = time.time() - self.start_time
        # steps_per_sec = self.local_t / elapsed_time
        # logging.info("### {}: Performance: {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(self.thread_index, self.local_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

        feed_dict = {
          self.local_network.s: [batch_si[0]],
          self.local_network.a: [batch_ai[0]],
          self.local_network.td: [batch_td[0]],
          self.local_network.r_duration: [batch_R_duration[0]],
          self.local_network.r_packets: [batch_R_packets[0]],
          self.local_network.r_accumulated_delay: [batch_R_accumulated_delay[0]],
          self.local_network.initial_lstm_state_action: self.start_lstm_states[0][0],
          self.local_network.initial_lstm_state_value: self.start_lstm_states[0][1],
          self.local_network.step_size : [1]
        }
        feed_dict.update(var_dict)

        entropy, actor_loss, value_loss, total_loss, window_increase, std = self.local_network.run_loss(sess, feed_dict)

        things = {
          # "score_throughput": normalized_final_score_throughput,
          "R_duration": batch_R_duration[0],
          "R_packets": batch_R_packets[0],
          "R_accumulated_delay": batch_R_accumulated_delay[0],
          "score_delay": normalized_final_score_delay,
          "actor_loss": actor_loss.item(),
          "value_loss": value_loss,
          "entropy": entropy.item(),
          "total_loss": total_loss,
          "window_increase": window_increase.item(),
          "window": windows[-1],
          "std": std.item(),
          # "speed": steps_per_sec
          }
        # logging.debug(" ".join(map(str,("things", things))))
        self._record_score(sess, summary_writer, summary_op, summary_inputs, things, global_t)

      if final:
        self.episode_count += 1
      # self.local_t = 0
      self.episode_reward_throughput = 0
      self.episode_reward_delay = 0

    self.restore_backup()

    self.actions = self.actions[LOCAL_T_MAX:]
    self.ticknos = self.ticknos[LOCAL_T_MAX:]
    self.windows = self.windows[LOCAL_T_MAX:]
    self.states = self.states[LOCAL_T_MAX:]
    self.values = self.values[LOCAL_T_MAX:]
    self.rewards = self.rewards[LOCAL_T_MAX:]
    self.durations = self.durations[LOCAL_T_MAX:]
    self.estimated_values = self.estimated_values[LOCAL_T_MAX:]
    self.time_differences = self.time_differences[1:]
    self.start_lstm_states = self.start_lstm_states[1:]
    self.variable_snapshots = self.variable_snapshots[1:]

    return diff_local_t