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
# gamma_current, gamma_future = 1./(1+GAMMA), GAMMA/(1.+GAMMA)
from constants import LOCAL_T_MAX
from constants import DELAY_MULTIPLIER
from constants import LOG_LEVEL

import logging
logging.basicConfig(level=LOG_LEVEL)

LOG_INTERVAL = 5

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device,
               training):

    self.training = training
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
      zip(self.gradients, global_network.get_vars()) )

    self.sync = self.local_network.sync_from(global_network)
    self.episode_count = 0

    self.backup_vars = self.local_network.backup_vars()
    self.restore_backup = self.local_network.restore_backup()

    self.initial_learning_rate = initial_learning_rate

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
      summary_inputs["skewness"]: things["skewness"],
      summary_inputs["actor_loss"]: things["actor_loss"],
      summary_inputs["value_loss"]: things["value_loss"],
      summary_inputs["total_loss"]: things["total_loss"],
      summary_inputs["window"]: things["window"],
      summary_inputs["std"]: things["std"],
      summary_inputs["inner_mean"]: things["inner_mean"],
      summary_inputs["inner_std"]: things["inner_std"],
      summary_inputs["speed"]: things["speed"]
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

  def set_start_time(self, start_time):
    self.start_time = start_time

  def action_step(self, sess, state):
    # Run this still with the old weights, before syncing them
    if self.training:
      self.estimated_values.append(self.local_network.run_value(sess, state))

      if len(self.actions) % LOCAL_T_MAX == 0:
        self.time_differences.append(None)
        # Sync for the next iteration
        sess.run( self.sync )
        self.start_lstm_states.append(self.local_network.lstm_state_out)
        self.variable_snapshots.append(sess.run(self.local_network.get_vars()))

    pi_, action, value_ = self.local_network.run_policy_action_and_value(sess, state)
    logging.debug(" ".join(map(str,(self.thread_index,"pi_values:",pi_))))

    if self.training:
      self.states.append(state)
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

    if len(self.rewards)>=LOCAL_T_MAX or (len(self.actions[:LOCAL_T_MAX]) > len([item for item in self.actions[:LOCAL_T_MAX] if item is not None]) and len([item for item in self.actions[:LOCAL_T_MAX] if item is not None]) == len(self.rewards)):
      assert(len(self.rewards) <= LOCAL_T_MAX)
      return self.process(sess, global_t, summary_writer, summary_op, summary_inputs)
    else:
      return 0

  def final_step(self, sess, global_t, summary_writer, summary_op, summary_inputs, actions_to_remove, time_difference):
    # self.actions = self.actions[:-actions_to_remove]
    # self.states = self.states[:-actions_to_remove]
    # self.values = self.values[:-actions_to_remove]
    # self.estimated_values = self.estimated_values[:-actions_to_remove+1]

    filling_nones = [None] * (LOCAL_T_MAX - (len(self.actions) % LOCAL_T_MAX))
    self.time_differences = self.time_differences[:-1]
    self.time_differences.append(time_difference)
    self.actions += filling_nones
    self.states += filling_nones
    self.values += filling_nones
    self.estimated_values += filling_nones

    # If, for some strange reason, absolutely nothing happened in this episode, don't do anyting...
    # Or if you're actually in testing mode :)
    time_diff = 0
    # if self.training:
    #   if len(self.rewards)>0:
    #     time_diff = self.process(sess, global_t, summary_writer, summary_op, summary_inputs, time_difference)

    self.local_network.reset_state()
    # sess.run( self.sync )

    return time_diff

  def process(self, sess, global_t, summary_writer, summary_op, summary_inputs):
    final = self.time_differences[0] is not None

    start_local_t = self.local_t

    # logging.debug(" ".join(map(str,(self.thread_index, "In process: len(rewards)", len(self.rewards), "len(durations)", len(self.durations), "len(states)", len(self.states), "len(actions)", len(self.actions), "len(values)", len(self.values)))))

    actions = [item for item in self.actions[:LOCAL_T_MAX] if item is not None]
    states = [item for item in self.states[:LOCAL_T_MAX] if item is not None]
    rewards = [item for item in self.rewards[:LOCAL_T_MAX] if item is not None]
    durations = [item for item in self.durations[:LOCAL_T_MAX] if item is not None]
    values = [item for item in self.values[:LOCAL_T_MAX] if item is not None]

    # logging.debug(" ".join(map(str,(self.thread_index, "In process: rewards", rewards, "durations", durations, "states", states, "actions", actions, "values", values))))

    diff_local_t = 0
    if len(actions) > 0:
      # get estimated value of step n+1
      assert((not len(self.estimated_values) <= len(rewards)) or final)
      if not final:
        # TODO: Try without the following if
        R_packets, R_accumulated_delay, R_duration = self.estimated_values[len(rewards)] if self.estimated_values[len(rewards)] is not None else self.estimated_values[len(rewards)-1]
      else:
        R_packets, R_accumulated_delay, R_duration = self.estimated_values[len(rewards)-1]

      logging.debug("final:"+str(final)+", "+" ".join(map(str,("R_packets", R_packets, "R_accumulated_delay", R_accumulated_delay, "R_duration", R_duration))))

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
      batch_td_throughput = []
      batch_td_delay = []
      batch_R_duration = []
      batch_R_packets = []
      batch_R_accumulated_delay = []

      # compute and accmulate gradients
      for(ai, ri, di, si, Vi) in zip(actions, rewards, durations, states, values):

        R_duration = (di + GAMMA*R_duration)

        R_packets = (ri[0] + GAMMA*R_packets)
        # TODO: In theory it should be np.log(R_bytes/R_duration) but remy doesn't have bytes
        td_throughput = (np.log(R_packets/R_duration) - np.log(Vi[0]/Vi[2]))

        R_accumulated_delay = (ri[1] + GAMMA*R_accumulated_delay)
        # R_delay = R_accumulated_delay/R_packets
        td_delay = -(np.log(R_accumulated_delay/R_packets / DELAY_MULTIPLIER) - np.log(Vi[1]/Vi[0] / DELAY_MULTIPLIER))

        batch_si.append(si)
        batch_ai.append(ai)
        batch_td_throughput.append(td_throughput)
        batch_td_delay.append(td_delay)
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

      # return advanced local step size
      diff_local_t = self.local_t - start_local_t

      cur_learning_rate = self._anneal_learning_rate(global_t)

      # logging.debug(" ".join(map(str,("All the batch stuff", "batch_si", batch_si, "batch_ai", batch_ai, "batch_td_throughput", batch_td_throughput, "batch_td_delay", batch_td_delay,"batch_R_packets", batch_R_packets, "batch_R_accumulated_delay", batch_R_accumulated_delay, "batch_R_duration", batch_R_duration))))

      self.backup_vars()

      batch_si.reverse()
      batch_ai.reverse()
      batch_td_throughput.reverse()
      batch_td_delay.reverse()
      batch_R_duration.reverse()
      batch_R_packets.reverse()
      batch_R_accumulated_delay.reverse()

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
        self.learning_rate_input: cur_learning_rate
      }
      var_dict = dict(zip(self.local_network.get_vars(), self.variable_snapshots[0]))
      feed_dict.update(var_dict)

      sess.run( self.apply_gradients,
                feed_dict = feed_dict )

    if final:
      if len(actions) > 0:
        if self.episode_count % LOG_INTERVAL == 0:
          normalized_final_score_throughput = self.episode_reward_throughput/self.time_differences[0]
          normalized_final_score_delay = self.episode_reward_delay/self.episode_reward_throughput
          logging.debug("{}: score_throughput={}, score_delay={}".format(self.thread_index, normalized_final_score_throughput, normalized_final_score_delay))

          feed_dict = {
            self.local_network.s: [batch_si[0]],
            self.local_network.a: [batch_ai[0]],
            self.local_network.td_throughput: [batch_td_throughput[0]],
            self.local_network.td_delay: [batch_td_delay[0]],
            self.local_network.r_duration: [batch_R_duration[0]],
            self.local_network.r_packets: [batch_R_packets[0]],
            self.local_network.r_accumulated_delay: [batch_R_accumulated_delay[0]],
            self.local_network.initial_lstm_state: self.start_lstm_states[0],
            self.local_network.step_size : [1],
          }
          feed_dict.update(var_dict)

          entropy, skewness, actor_loss, value_loss, total_loss, window, std, inner_mean, inner_std = self.local_network.run_loss(sess, feed_dict)

          elapsed_time = time.time() - self.start_time
          steps_per_sec = self.local_t / elapsed_time
          logging.info("### {}: Performance: {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(self.thread_index, self.local_t, elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

          things = {"score_throughput": normalized_final_score_throughput,
            "score_delay": normalized_final_score_delay,
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss,
            "entropy": entropy.item(),
            "skewness": skewness.item(),
            "total_loss": total_loss,
            "window": window.item(),
            "std": std.item(),
            "inner_mean": inner_mean.item(),
            "inner_std": inner_std.item(),
            "speed": steps_per_sec}
          logging.debug(" ".join(map(str,("things", things))))
          self._record_score(sess, summary_writer, summary_op, summary_inputs, things, global_t)

      self.local_t = 0
      self.episode_reward_throughput = 0
      self.episode_reward_delay = 0

      self.episode_count += 1
    else:
      self.restore_backup()

    self.actions = self.actions[LOCAL_T_MAX:]
    self.states = self.states[LOCAL_T_MAX:]
    self.values = self.values[LOCAL_T_MAX:]
    self.estimated_values = self.estimated_values[LOCAL_T_MAX:]
    self.rewards = []
    self.durations = []
    self.start_lstm_states = self.start_lstm_states[1:]
    self.variable_snapshots = self.variable_snapshots[1:]
    self.time_differences = self.time_differences[1:]

    return diff_local_t

