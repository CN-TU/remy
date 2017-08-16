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

from game_ac_network import GameACLSTMNetwork#, GameACFFNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import USE_LSTM
from constants import ACTION_SIZE
from constants import ALPHA
from constants import BETA

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

def sigmoid(x):
  return 1.0 / (1. + np.exp(-x))

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
    self.prev_local_t = 0

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

  # def _accumulate(self, state):
  #   state = np.array(state)
  #   if self.acc_state is None:
  #     self.acc_state = np.stack((state, state, state, state), axis=1)
  #   else:
  #     # print(self.acc_state)
  #     state = np.expand_dims(state, 1)
  #     # print(state)
  #     self.acc_state = np.append(self.acc_state[:,1:], state, axis=1)
  #   return self.acc_state.flatten()

  def action_step(self, sess, state):

    if len(self.actions) % LOCAL_T_MAX == 0:
      # Sync for the next iteration
      sess.run( self.sync )
      if USE_LSTM:
        self.start_lstm_states.append(self.local_network.lstm_state_out)

    pi_, action, value_ = self.local_network.run_policy_action_and_value(sess, state)
    print("pi_values:",pi_)
    # action = self.choose_action(pi_)

    self.states.append(state)

    self.actions.append(np.copy(action))

    action = np.copy(action)
    action[0] = max(action[0], 1.0)

    self.values.append(value_)
    # if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
    if self.local_t % LOG_INTERVAL == 0:
      print("pi={}".format(pi_))
      print(" V={}".format(value_))
    return (action[0], 0, 0)

  def reward_step(self, sess, global_t, summary_writer, summary_op, summary_inputs, reward_throughput, reward_delay, duration):
    self.rewards.append((reward_throughput, reward_delay))
    self.durations.append(duration)

    # There must be always at least 1 item left in the next batch:
    # one to be removed and one to finish the sequence. 
    if len(self.rewards)-1>=LOCAL_T_MAX:
      return self.process(sess, global_t, summary_writer, summary_op, summary_inputs)
    # implicitly returns None otherwise

  def final_step(self, sess, global_t, summary_writer, summary_op, summary_inputs, final, actions_to_remove):
    self.actions = self.actions[:-actions_to_remove]
    self.states = self.states[:-actions_to_remove]
    self.values = self.values[:-actions_to_remove]

    if len(self.rewards) > 0:
      return self.process(sess, global_t, summary_writer, summary_op, summary_inputs, final)
    else:
      return 0

  def process(self, sess, global_t, summary_writer, summary_op, summary_inputs, final=False):

    start_local_t = self.local_t
    
    print(self.thread_index, "In process: len(rewards)", len(self.rewards), "len(durations)", len(self.durations), "len(states)", len(self.states), "len(actions)", len(self.actions), "len(values)", len(self.values))

    actions = self.actions[:LOCAL_T_MAX]
    states = self.states[:LOCAL_T_MAX]
    rewards = self.rewards[:LOCAL_T_MAX]
    durations = self.durations[:LOCAL_T_MAX]
    values = self.values[:LOCAL_T_MAX]

    print(self.thread_index, "In process: rewards", rewards, "durations", durations, "states", states, "actions", actions, "values", values)

    # t_max times loop
    for i in range(len(rewards)):
      self.episode_reward_throughput += rewards[i][0]*ALPHA
      self.episode_reward_delay += rewards[i][1]*BETA

      # clip reward
      # rewards[i] = np.clip(rewards[i], -1, 1) ) # TODO: Why do this? Is the range from -1 to 1 problem-specific?

      self.local_t += 1

    if final:
      # R_throughput = R_delay = 0.0
      # FIXME: This is a hack! It should evaluate how good the very last state is after the last action
      R_throughput, R_delay = self.local_network.run_value(sess, self.states[-1])
      print("R_throughput, R_delay", R_throughput, R_delay)
    else:
      R_throughput, R_delay = self.local_network.run_value(sess, self.states[LOCAL_T_MAX])
      print("R_throughput, R_delay", R_throughput, R_delay)

    # R_throughput = R_delay = 0.0
    R_throughput = np.exp(R_throughput)
    R_delay = np.exp(-R_delay)*R_throughput
    print("exp(R_throughput), exp(R_delay)", R_throughput, R_delay)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()
    # print("values", values)

    batch_si = []
    batch_ai = []
    batch_td_throughput = []
    batch_td_delay = []
    batch_R_throughput = []
    batch_R_delay = []

    # compute and accmulate gradients
    for(ai, ri, di, si, Vi) in zip(actions, rewards, durations, states, values):
      # print("ri[0]/di + GAMMA * R_throughput", ri[0]/di, GAMMA, R_throughput)
      R_throughput = ri[0]*ALPHA + GAMMA * R_throughput
      td_throughput = np.log(R_throughput) - Vi[0]

      # print("(ri[1]/ri[0]) + GAMMA * R_delay", average_delay, GAMMA, R_delay)
      R_delay = ri[1] + GAMMA * R_delay
      td_delay = -np.log(R_delay/R_throughput) - Vi[1]

      batch_si.append(si)
      batch_ai.append(ai)
      batch_td_throughput.append(td_throughput)
      batch_td_delay.append(td_delay)
      batch_R_throughput.append(np.log(R_throughput))
      batch_R_delay.append(-np.log(R_delay/R_throughput))

      print("batch_td_throughput[-1]", batch_td_throughput[-1], "batch_td_delay[-1]", batch_td_delay[-1], "batch_R_throughput[-1]", batch_R_throughput[-1], "batch_R_delay[-1]", batch_R_delay[-1])

      assert(np.isfinite(batch_td_throughput[-1]))
      assert(np.isfinite(batch_td_delay[-1]))
      assert(np.isfinite(batch_R_throughput[-1]))
      assert(np.isfinite(batch_R_delay[-1]))

    # print(self.thread_index, "Got the following rewards:", rewards, "values", values, "R", R)
    cur_learning_rate = self._anneal_learning_rate(global_t)
    # print(self.thread_index, "Still alive!", cur_learning_rate)

    print("All the batch stuff", "batch_si", batch_si, "batch_ai", batch_ai, "batch_td_throughput", batch_td_throughput, "batch_td_delay", batch_td_delay,"batch_R_throughput", batch_R_throughput, "batch_R_delay", batch_R_delay)

    if USE_LSTM:
      batch_si.reverse()
      batch_ai.reverse()
      batch_td_throughput.reverse()
      batch_td_delay.reverse()
      batch_R_throughput.reverse()
      batch_R_delay.reverse()

      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_ai,
                  self.local_network.td_throughput: batch_td_throughput,
                  self.local_network.td_delay: batch_td_delay,
                  self.local_network.r_throughput: batch_R_throughput,
                  self.local_network.r_delay: batch_R_delay,
                  self.local_network.initial_lstm_state: self.start_lstm_states[0],
                  self.local_network.step_size : [len(batch_ai)],
                  self.learning_rate_input: cur_learning_rate } )
    else:
      raise NotImplementedError("FF currently not implemented.")
      # print("learning_rate_input", cur_learning_rate)
      # sess.run( self.apply_gradients,
      #           feed_dict = {
      #             self.local_network.s: batch_si,
      #             self.local_network.a: batch_ai,
      #             self.local_network.td: batch_td,
      #             self.local_network.r: batch_R,
      #             self.learning_rate_input: cur_learning_rate} )
      
    # if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
    if self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL:
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    if final:
      normalized_final_score_throughput = self.episode_reward_throughput/(self.local_t-self.episode_start_t)
      normalized_final_score_delay = self.episode_reward_delay/self.episode_reward_throughput
      print("score_throughput={}, score_delay={}".format(normalized_final_score_throughput, normalized_final_score_delay))
      entropy, action_loss, value_loss, total_loss, window, std = self.local_network.run_loss(sess, batch_si[-1], batch_ai[-1], batch_td_throughput[-1], batch_td_delay[-1], batch_R_throughput[-1], batch_R_delay[-1])
      things = {"score_throughput": normalized_final_score_throughput, 
        "score_delay": normalized_final_score_delay, 
        "action_loss": action_loss.item(),
        "value_loss": value_loss,
        "entropy": entropy.item(),
        "total_loss": total_loss,
        "window": window.item(),
        "std": std.item()}
      print("things", things)
      self._record_score(sess, summary_writer, summary_op, summary_inputs, things, global_t) # TODO:NOW: is that "not terminal_end" correct?
      self.episode_start_t = self.local_t
      self.episode_reward_throughput = 0
      self.episode_reward_delay = 0
      if USE_LSTM:
        self.local_network.reset_state()

    self.actions = self.actions[LOCAL_T_MAX:]
    self.states = self.states[LOCAL_T_MAX:]
    self.values = self.values[LOCAL_T_MAX:]
    self.rewards = self.rewards[LOCAL_T_MAX:]
    self.durations = self.durations[LOCAL_T_MAX:]
    self.start_lstm_states = self.start_lstm_states[1:]
    # return advanced local step size
    diff_local_t = self.local_t - start_local_t

    return diff_local_t
    
