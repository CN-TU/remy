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
from functools import reduce

from game_ac_network import GameACLSTMNetwork

# from constants import GAMMA
from constants import GAMMA_FACTOR
# gamma_current, gamma_future = 1./(1+GAMMA), GAMMA/(1.+GAMMA)
from constants import LOCAL_T_MAX
from constants import LOG_LEVEL
from constants import SECONDS_NORMALIZER
from constants import STATE_SIZE
from constants import SIGMOID_ALPHA
from constants import MAX_WINDOW

import logging

logging.basicConfig(level=LOG_LEVEL)

LOG_INTERVAL = 200
# LOG_INTERVAL = 5

def inverse_sigmoid(x):
  return 1/(1+math.exp(SIGMOID_ALPHA*x))

# def inverse_sigmoid(x):
#   return min(1-x, 1)

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

  def reset_state_and_reinitialize(self, sess):
    self.local_network.reset_state()
    # action, value_ = self.local_network.run_action_and_value(sess, [0.0]*STATE_SIZE)

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

  def start_anew(self):
    assert(len(self.windows) > 0)

    current_index = 0
    while current_index < len(self.windows):
      if current_index + math.floor(A3CTrainingThread.get_actual_window(self.windows[current_index]+self.actions[current_index])) == len(self.windows):
        return True
      current_index = current_index + math.floor(A3CTrainingThread.get_actual_window(self.windows[current_index]+self.actions[current_index]))
    return False

  def action_step(self, sess, state, tickno, window):
    # print(self.thread_index, "in action")
    # Run this still with the old weights, before syncing them
    # print("state", state)
    assert(np.all(np.isfinite(np.array(state, dtype=np.float32))))

    # print(self.thread_index, "state", state)
    if self.training:
      self.estimated_values.append(self.local_network.run_value(sess, state))

      # if len(self.actions) % LOCAL_T_MAX == 0:
      if not (len(self.start_lstm_states) == 0) == (len(self.actions) == 0):
        print("Oh no, something went pretty wrong:", self.start_lstm_states, self.actions)
      assert((len(self.start_lstm_states) == 0) == (len(self.actions) == 0))
      if ('LOCAL_T_MAX' in globals() and len(self.actions) % globals()["LOCAL_T_MAX"] == 0) or (not 'LOCAL_T_MAX' in globals() and (len(self.actions) == 0 or self.start_anew())):
        # print("Starting new period")
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
    # print(self.thread_index, action[0])
    return action

  def reward_step(self, sess, global_t, summary_writer, summary_op, summary_inputs, reward_throughput, reward_delay, duration, sent):
    # print(self.thread_index, "in reward")
    assert(reward_throughput >= 0)
    assert(reward_delay >= 0)
    # print("duration", duration)
    # assert(duration > 0)
    assert(sent>= 0)

    # assert(sent == reward_throughput)
    self.rewards.append((reward_throughput, reward_delay, duration, sent))

    # if len(self.rewards)>=LOCAL_T_MAX or (len([item for item in self.actions[:LOCAL_T_MAX] if item is not None]) == len(self.rewards) and len(self.rewards) > 0 and self.time_differences[0] is not None):
    if ('LOCAL_T_MAX' in globals() and (len(self.rewards)>=globals()["LOCAL_T_MAX"] or (len([item for item in self.actions[:globals()["LOCAL_T_MAX"]] if item is not None]) == len(self.rewards) and len(self.rewards) > 0 and self.time_differences[0] is not None))) or (not 'LOCAL_T_MAX' in globals() and len(self.rewards)>=math.floor(A3CTrainingThread.get_actual_window(self.windows[0]+self.actions[0]))):
      if not 'LOCAL_T_MAX' in globals():
        assert(len(self.rewards) == math.floor(A3CTrainingThread.get_actual_window(self.windows[0]+self.actions[0])))
      # print(self.thread_index, "rewards", self.rewards, "actions", self.actions, "time_diffs", self.time_differences)
      # assert(len(self.rewards) <= LOCAL_T_MAX)
      # print(len([item for item in self.actions[:LOCAL_T_MAX] if item is not None]), len(self.rewards[:LOCAL_T_MAX]))

      # assert(len([item for item in self.actions[:LOCAL_T_MAX] if item is not None]) == len(self.rewards[:LOCAL_T_MAX]))
      assert(len([item for item in self.actions[:len(self.rewards)] if item is not None]) == len(self.rewards[:len(self.rewards)]))
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

        if 'LOCAL_T_MAX' in globals():
          nones_to_add = [None] * ((LOCAL_T_MAX - (len(self.actions) % LOCAL_T_MAX)) % LOCAL_T_MAX)
          self.actions += nones_to_add
          self.states += nones_to_add
          self.values += nones_to_add
          self.estimated_values += nones_to_add
          self.windows += nones_to_add
          self.ticknos += nones_to_add
      # TODO: Is this useful? I guess only the `local_t' is actually needed...
      else:
        self.local_t = 0
        self.episode_count += 1
        self.episode_reward_throughput = 0
        self.episode_reward_delay = 0
        self.episode_reward_sent = 0

    # If, for some strange reason, absolutely nothing happened in this episode, don't do anything...
    # Or if you're actually in testing mode :)
    # if len(self.rewards)>0:
    #   time_diff = self.process(sess, global_t, summary_writer, summary_op, summary_inputs, time_difference)
    # else:
    #   time_diff = 0

    # self.states = []
    # self.actions = []
    # self.rewards = []
    # self.values = []
    # self.estimated_values = []
    # self.start_lstm_states = []
    # self.variable_snapshots = []
    # FIXME: Not resetting state any longer!!! Is that bad?
    sess.run( self.sync )
    self.reset_state_and_reinitialize(sess)

  @staticmethod
  def get_actual_window(x):
    return max(min(x, MAX_WINDOW), 1)

  def process(self, sess, global_t, summary_writer, summary_op, summary_inputs, time_difference=None):
    # print(self.thread_index, "in process")
    assert(len(self.rewards) > 0)

    if not len(self.start_lstm_states) <= len(self.actions):
      print(len(self.start_lstm_states), len(self.actions))
    assert(len(self.start_lstm_states) <= len(self.actions))

    if self.local_t <= 0:
      self.start_time = time.time()

    final = time_difference is not None

    start_local_t = self.local_t

    # logging.debug(" ".join(map(str,(self.thread_index, "In process: len(rewards)", len(self.rewards), "len(states)", len(self.states), "len(actions)", len(self.actions), "len(values)", len(self.values)))))

    if 'LOCAL_T_MAX' in globals():
      actions = [item for item in self.actions[:LOCAL_T_MAX] if item is not None]
      ticknos = [item for item in self.ticknos[:LOCAL_T_MAX] if item is not None]
      windows = [item for item in self.windows[:LOCAL_T_MAX] if item is not None]
      states = [item for item in self.states[:LOCAL_T_MAX] if item is not None]
      rewards = [item for item in self.rewards[:LOCAL_T_MAX] if item is not None]
      values = [item for item in self.values[:LOCAL_T_MAX] if item is not None]
    else:
      actions = self.actions[:len(self.rewards)]
      ticknos = self.ticknos[:len(self.rewards)]
      windows = self.windows[:len(self.rewards)]
      states = self.states[:len(self.rewards)]
      values = self.values[:len(self.rewards)]
      rewards = self.rewards[:len(self.rewards)]

    assert(len(actions) > 0)
    assert(len(ticknos) > 0)
    assert(len(windows) > 0)
    assert(len(states) > 0)
    assert(len(rewards) > 0)
    assert(len(values) > 0)
    if not (len(actions) == len(ticknos) == len(windows) == len(states) == len(rewards) == len(values)):
      print(len(self.actions), len(self.ticknos), len(self.windows), len(self.states), len(self.rewards), len(self.values))
      print(len(actions), len(ticknos), len(windows), len(states), len(rewards), len(values))
      print(self.actions, self.ticknos, self.windows, self.states, self.rewards, self.values)
      print(actions, ticknos, windows, states, rewards, values)
    assert(len(actions) == len(ticknos) == len(windows) == len(states) == len(rewards) == len(values))

    # if not len(self.actions) == len(self.ticknos) == len(self.windows) == len(self.states) == len(self.values) == len(self.estimated_values):
    #   print("In thread:", self.thread_index, "rewards:", len(self.rewards), ";", len(self.actions), len(self.ticknos), len(self.windows), len(self.states), len(self.values), len(self.estimated_values))
    # print("In thread:", self.thread_index, "rewards:", len(self.rewards), ";", len(self.actions), "lstm_states:", len(self.start_lstm_states))
    assert(len(self.actions) == len(self.ticknos) == len(self.windows) == len(self.states) == len(self.values) == len(self.estimated_values))
    assert(len(self.time_differences) == len(self.start_lstm_states) == len(self.variable_snapshots))

    # logging.debug(" ".join(map(str,(self.thread_index, "In process: rewards", rewards, "states", states, "actions", actions, "values", values))))

    # get estimated value of step n+1
    # assert((not len(self.estimated_values) <= len(rewards)) or final)
    # print("self.estimated_values", self.estimated_values)
    # print("Spam and eggs")
    R_packets, R_accumulated_delay, R_duration, R_sent = self.estimated_values[len(rewards)] if self.estimated_values[len(rewards)] is not None and not final else self.estimated_values[len(rewards)-1]

    R_packets_initial, R_accumulated_delay_initial, R_duration_initial, R_sent_initial = R_packets, R_accumulated_delay, R_duration, R_sent

    # R_packets, R_accumulated_delay, R_duration, R_sent = (R_packets)/(1-GAMMA), (R_accumulated_delay)/(1-GAMMA), (R_duration)/(1-GAMMA), (R_sent)/(1-GAMMA)
    # logging.debug(" ".join(map(str,("exp(R_packets)", R_packets, "exp(R_accumulated_delay)", R_accumulated_delay, "exp(R_duration)", R_duration))))
    assert(np.isfinite(R_duration))
    assert(np.isfinite(R_packets))
    assert(np.isfinite(R_accumulated_delay))
    assert(np.isfinite(R_sent))

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()
    windows.reverse()
    # logging.debug(" ".join(map(str,("values", values))))

    batch_si = []
    batch_ai = []
    batch_td = []
    batch_R_duration = []
    batch_R_packets = []
    batch_R_accumulated_delay = []
    batch_R_sent = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi, wi) in zip(actions, rewards, states, values, windows):
      # FIXME: Make sure that it actually works with how the roll-off factor gets normalized.
      # assert(False)
      # The GAMMA_FACTOR increases the influence that following observations have on this one.
      GAMMA = (1 - 2/(A3CTrainingThread.get_actual_window(wi+ai) + 2))

      # R_duration = ((1-GAMMA)*ri[2]*SECONDS_NORMALIZER + GAMMA*R_duration)
      # R_packets = ((1-GAMMA)*ri[0] + GAMMA*R_packets)
      # R_sent = ((1-GAMMA)*ri[3] + GAMMA*R_sent)
      # R_accumulated_delay = ((1-GAMMA)*ri[1]*SECONDS_NORMALIZER + GAMMA*R_accumulated_delay)

      R_duration = ((1-GAMMA)*ri[2]*SECONDS_NORMALIZER + GAMMA*R_duration)
      R_packets = ((1-GAMMA)*ri[0] + GAMMA*R_packets)
      R_sent = ((1-GAMMA)*ri[3] + GAMMA*R_sent)
      R_accumulated_delay = ((1-GAMMA)*ri[1]*SECONDS_NORMALIZER + GAMMA*R_accumulated_delay)

      # R_delay = R_accumulated_delay/R_packets
      # td_delay = -(np.log(R_accumulated_delay/R_packets/DELAY_MULTIPLIER) - np.log(Vi[1]/Vi[0]/DELAY_MULTIPLIER))
      # td -= self.delay_delta/SECONDS_NORMALIZER*(R_accumulated_delay/R_packets - Vi[1]/Vi[0])
      # td_delay = -(R_accumulated_delay/R_packets - Vi[1]/Vi[0])

      # Doesn't work...
      # td = R_packets/R_duration/(R_accumulated_delay/R_packets+self.delay_delta) - Vi[0]/Vi[2]/(Vi[1]/Vi[0]+self.delay_delta)

      # td = R_packets - Vi[0] - self.delay_delta*(R_sent - Vi[3]) # - self.delay_delta*(R_accumulated_delay/R_packets - Vi[1]/Vi[0])
      # td = R_packets/R_duration - Vi[0]/Vi[2] - self.delay_delta*(R_accumulated_delay/R_packets - Vi[1]/Vi[0])
      # td = R_packets/R_duration - Vi[0]/(Vi[2]) - self.delay_delta*(R_accumulated_delay/R_packets - Vi[1]/Vi[0]) - (R_sent/R_duration - Vi[3]/(Vi[2]))
      # td = inverse_sigmoid(self.delay_delta, R_sent/(R_packets+R_sent))*R_packets/R_duration - inverse_sigmoid(self.delay_delta, Vi[3]/(Vi[0]+Vi[3]))*Vi[0]/Vi[2] - (R_sent/R_duration - Vi[3]/(Vi[2]))

      # td = inverse_sigmoid(self.delay_delta, R_sent/(R_packets+R_sent) - 0.05)*R_packets/R_duration - inverse_sigmoid(self.delay_delta, Vi[3]/(Vi[0]+Vi[3]) - 0.05)*Vi[0]/Vi[2] - (R_sent/R_duration - Vi[3]/(Vi[2])) - (R_accumulated_delay/R_packets - Vi[1]/Vi[0])

      # PCC
      # td = inverse_sigmoid(((R_sent - R_packets)/R_sent) - self.delay_delta)*R_packets/R_duration - inverse_sigmoid(((Vi[3]-Vi[0])/Vi[3]) - self.delay_delta)*Vi[0]/Vi[2] - ((R_sent - R_packets)/R_duration - (Vi[3] - Vi[0])/Vi[2]) #- (R_accumulated_delay/R_packets - Vi[1]/Vi[0])

      # td = R_packets*(1-GAMMA)*inverse_sigmoid(SIGMOID_ALPHA * R_sent/(R_packets+R_sent) - self.delay_delta) - Vi[0]*inverse_sigmoid(SIGMOID_ALPHA * Vi[3]/(Vi[0]+Vi[3]) - self.delay_delta) - (R_sent*(1-GAMMA) - Vi[3]) - (R_accumulated_delay/R_packets - Vi[1]/Vi[0])

      # PCC modified
      # td = R_packets*inverse_sigmoid(SIGMOID_ALPHA * R_sent/(R_packets+R_sent) - self.delay_delta) - Vi[0]*inverse_sigmoid(SIGMOID_ALPHA * Vi[3]/(Vi[0]+Vi[3]) - self.delay_delta) - (R_sent - Vi[3])# - (R_accumulated_delay/R_packets - Vi[1]/Vi[0])

      # td = R_packets/R_duration/(R_accumulated_delay/R_packets) - Vi[0]/Vi[2]/(Vi[1]/Vi[0]) - (R_accumulated_delay/R_packets - Vi[1]/Vi[0])


      td = (np.log(R_packets/R_duration) - np.log(Vi[0]/Vi[2])) - self.delay_delta/SECONDS_NORMALIZER*(R_accumulated_delay/R_packets - Vi[1]/Vi[0])


      # R_packets, R_accumulated_delay, R_duration, R_sent = (R_packets)/(1-GAMMA), (R_accumulated_delay)/(1-GAMMA), (R_duration)/(1-GAMMA), (R_sent)/(1-GAMMA)

      batch_si.append(si)
      batch_ai.append(ai)
      batch_td.append(td)
      batch_R_duration.append(R_duration)
      batch_R_packets.append(R_packets)
      batch_R_accumulated_delay.append(R_accumulated_delay)
      batch_R_sent.append(R_sent)

      # batch_R_duration.append(R_duration/(1-GAMMA))
      # batch_R_packets.append(R_packets/(1-GAMMA))
      # batch_R_accumulated_delay.append(R_accumulated_delay/(1-GAMMA))
      # batch_R_sent.append(R_sent/(1-GAMMA))

      # logging.debug(" ".join(map(str,("batch_td_throughput[-1]", batch_td_throughput[-1], "batch_td_delay[-1]", batch_td_delay[-1], "batch_R_packets[-1]", batch_R_packets[-1], "batch_R_accumulated_delay[-1]", batch_R_accumulated_delay[-1], "batch_R_duration[-1]", batch_R_duration[-1]))))

      self.episode_reward_throughput += ri[0]
      self.episode_reward_sent += ri[3]
      self.episode_reward_delay += ri[1]

    old_local_t = self.local_t
    self.local_t += len(rewards)

    # if final or self.local_t % LOG_INTERVAL == 0:
    #   print(self.thread_index, "actions", len(self.actions), "rewards", len(self.rewards), "values", len(self.values), "estimated_values", len(self.estimated_values))
    #   print(self.thread_index, "actions", self.actions, "rewards", self.rewards, "values", self.values, "estimated_values", self.estimated_values)

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t

    cur_learning_rate = self._anneal_learning_rate(global_t)

    # logging.info(" ".join(map(str,("All the batch stuff", "batch_si", batch_si, "batch_ai", batch_ai,"batch_R_packets", batch_R_packets, "batch_R_accumulated_delay", batch_R_accumulated_delay, "batch_R_duration", batch_R_duration))))

    self.backup_vars()

    batch_si.reverse()
    batch_ai.reverse()
    batch_td.reverse()
    batch_R_duration.reverse()
    batch_R_packets.reverse()
    batch_R_accumulated_delay.reverse()
    batch_R_sent.reverse()
    windows.reverse()

    feed_dict = {
      self.local_network.s: batch_si,
      self.local_network.a: batch_ai,
      self.local_network.td: batch_td,
      self.local_network.r_duration: batch_R_duration,
      self.local_network.r_packets: batch_R_packets,
      self.local_network.r_accumulated_delay: batch_R_accumulated_delay,
      self.local_network.r_sent: batch_R_sent,
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
    if final or (self.local_t >= math.floor(self.local_t/LOG_INTERVAL)*LOG_INTERVAL and old_local_t < math.floor(self.local_t/LOG_INTERVAL)*LOG_INTERVAL):
    # if final:
      # if ticknos[-1]-ticknos[0] > 0 and self.episode_reward_throughput > 0:
      if self.episode_reward_throughput > 0:
        # print(ticknos)
        # print(self.episode_reward_throughput, ticknos[0], ticknos[-1])
        # normalized_final_score_throughput = self.episode_reward_throughput/(ticknos[-1]-ticknos[0])
        # logging.info("{}: self.episode_reward_throughput={}, time_difference={}".format(self.thread_index, self.episode_reward_throughput, time_difference))
        normalized_final_score_delay = self.episode_reward_delay/self.episode_reward_throughput
        loss_score = (self.episode_reward_sent - self.episode_reward_throughput)/self.episode_reward_sent
        # print(self.windows)

        # logging.info("{}: score_throughput={}, score_delay={}, measured throughput beginning={}, measured delay beginning={}, measured throughput end={}, measured delay end={}".format(self.thread_index, normalized_final_score_throughput, normalized_final_score_delay, batch_R_packets[0]/batch_R_duration[0]*SECONDS_NORMALIZER, batch_R_accumulated_delay[0]/batch_R_packets[0]/SECONDS_NORMALIZER, batch_R_packets[-1]/batch_R_duration[-1]*SECONDS_NORMALIZER, batch_R_accumulated_delay[-1]/batch_R_packets[-1]/SECONDS_NORMALIZER))
        # logging.info("{}: score_delay={}, measured throughput beginning={}, measured delay beginning={}, measured throughput end={}, measured delay end={} {}".format(self.thread_index, normalized_final_score_delay, batch_R_packets[0]/batch_R_duration[0]*SECONDS_NORMALIZER, batch_R_accumulated_delay[0]/batch_R_packets[0]/SECONDS_NORMALIZER, batch_R_packets[-1]/batch_R_duration[-1]*SECONDS_NORMALIZER, batch_R_accumulated_delay[-1]/batch_R_packets[-1]/SECONDS_NORMALIZER, ("final:"+str(final)+", delta:"+str(self.delay_delta)+"; "+" ".join(map(str,("R_packets", R_packets_initial, "R_accumulated_delay", R_accumulated_delay_initial/SECONDS_NORMALIZER, "R_duration", R_duration_initial/SECONDS_NORMALIZER))), "state", batch_si[0], "action", batch_ai[0][0])))
        logging.info("{}: score_delay={}, measured throughput beginning={}, measured delay beginning={} {}".format(self.thread_index, normalized_final_score_delay, batch_R_packets[0]/batch_R_duration[0]*SECONDS_NORMALIZER, batch_R_accumulated_delay[0]/batch_R_packets[0]/SECONDS_NORMALIZER, ("final:"+str(final)+", delta:"+str(self.delay_delta)+"; "+" ".join(map(str,("R_packets", R_packets_initial, "R_accumulated_delay", R_accumulated_delay_initial/SECONDS_NORMALIZER, "R_duration", R_duration_initial/SECONDS_NORMALIZER, "R_sent", R_sent_initial))), "state", batch_si[0], "action", batch_ai[0])))


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
          self.local_network.r_sent: [batch_R_sent[0]],
          self.local_network.initial_lstm_state_action: self.start_lstm_states[0][0],
          self.local_network.initial_lstm_state_value: self.start_lstm_states[0][1],
          self.local_network.step_size : [1]
        }
        feed_dict.update(var_dict)

        entropy, actor_loss, value_loss, total_loss, window_increase, std = self.local_network.run_loss(sess, feed_dict)

        things = {
          # "score_throughput": normalized_final_score_throughput,
          "estimated_throughput": batch_R_packets[0]/batch_R_duration[0]*SECONDS_NORMALIZER,
          "estimated_delay": batch_R_accumulated_delay[0]/batch_R_packets[0]/SECONDS_NORMALIZER,
          "estimated_loss_rate": (batch_R_sent[0] - batch_R_packets[0])/batch_R_sent[0],
          "R_duration": batch_R_duration[0]/SECONDS_NORMALIZER,
          "R_packets": batch_R_packets[0],
          "R_accumulated_delay": batch_R_accumulated_delay[0]/SECONDS_NORMALIZER,
          "R_sent": batch_R_sent[0],
          "score_delay": normalized_final_score_delay,
          "score_lost": loss_score,
          "actor_loss": actor_loss.item(),
          "value_loss": value_loss,
          "entropy": entropy.item(),
          "total_loss": total_loss,
          "window_increase": window_increase.item(),
          "window": windows[0],
          "std": std.item(),
          # "speed": steps_per_sec
          }
        # logging.debug(" ".join(map(str,("things", things))))
        self._record_score(sess, summary_writer, summary_op, summary_inputs, things, ticknos[0])

      # if final:
      self.episode_count += 1
      self.local_t = 0
      self.episode_reward_throughput = 0
      self.episode_reward_delay = 0
      self.episode_reward_sent = 0

    self.restore_backup()

    if 'LOCAL_T_MAX' in globals():
      self.actions = self.actions[LOCAL_T_MAX:]
      self.ticknos = self.ticknos[LOCAL_T_MAX:]
      self.windows = self.windows[LOCAL_T_MAX:]
      self.states = self.states[LOCAL_T_MAX:]
      self.values = self.values[LOCAL_T_MAX:]
      self.rewards = self.rewards[LOCAL_T_MAX:]
      self.estimated_values = self.estimated_values[LOCAL_T_MAX:]
    else:
      self.actions = self.actions[len(rewards):]
      self.ticknos = self.ticknos[len(rewards):]
      self.windows = self.windows[len(rewards):]
      self.states = self.states[len(rewards):]
      self.values = self.values[len(rewards):]
      self.estimated_values = self.estimated_values[len(rewards):]
      self.rewards = self.rewards[len(rewards):]
    self.time_differences = self.time_differences[1:]
    self.start_lstm_states = self.start_lstm_states[1:]
    self.variable_snapshots = self.variable_snapshots[1:]

    if final:
      assert(len(self.rewards) <= 0)

    return diff_local_t
