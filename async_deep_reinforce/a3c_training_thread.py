# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy as sp
from scipy.stats import norm, gamma
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
    self.values = []

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

    self.episode_reward = 0

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

  def choose_action(self, pi_values):
    print("choose action pi_values", pi_values)
    actions = [norm.rvs(loc=mean, scale=np.sqrt(var)) for mean, var in zip(pi_values[0], pi_values[1])]
    # actions[0] = max(0, actions[0]) # Make sure that multiplier isn't negative. There are no negative congestion windows. TODO: Maybe it would be a good idea?
    # actions[0] = math.log(1+math.exp(actions[0].item())) # Make sure that the minimum time to wait until you send the next packet isn't negative. 
    actions[0] = actions[0].item()
    # print("action", actions[0])
    return actions

  def _record_score(self, sess, summary_writer, summary_op, summary_inputs, things, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      summary_inputs["score"]: things["score"],
      summary_inputs["entropy"]: things["entropy"],
      summary_inputs["action_loss"]: things["action_loss"],
      summary_inputs["value_loss"]: things["value_loss"],
      summary_inputs["total_loss"]: things["total_loss"]
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

    # state = self._accumulate(state)
    # pi_, value_ = self.local_network.run_policy_and_value(sess, state)
    pi_, value_ = self.local_network.run_policy_and_value(sess, state)
    # assert(action[0] > 0.0)
    # print("action_step", pi_)
    action = self.choose_action(pi_)
    # This whole accumulation thing is just a big hack that is also used in the game implementation. Fortunately with LSTM it's not needed anymore hopefully. 

    self.states.append(state)
    self.actions.append(action)

    self.values.append(value_)
    # if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
    if self.local_t % LOG_INTERVAL == 0:
      print("pi={}".format(pi_))
      print(" V={}".format(value_))
    return (0, 0, action[0])

  def reward_step(self, sess, global_t, summary_writer, summary_op, summary_inputs, reward):
    self.rewards.append(reward)

    # There must be always at least 1 item left in the next batch:
    # one to be removed and one to finish the sequence. 
    if len(self.rewards)-1>=LOCAL_T_MAX:
      return self.process(sess, global_t, summary_writer, summary_op, summary_inputs)
    # implicitly returns None otherwise

  def final_step(self, sess, global_t, summary_writer, summary_op, summary_inputs, final):
    # if remove_last:
    #   self.actions = self.actions[:-1]
    #   self.states = self.states[:-1]
    #   # self.rewards = self.rewards[:-1]
    #   self.values = self.values[:-1]
    # if len(self.rewards) > 0:
    #   final_output = self.process(sess, global_t, summary_writer, summary_op, summary_inputs, final)
    # else:
    #   final_output = 0
    # return final_output
    return self.process(sess, global_t, summary_writer, summary_op, summary_inputs, final)

  def process(self, sess, global_t, summary_writer, summary_op, summary_inputs, final=False):
    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    if USE_LSTM:
      start_lstm_state = self.local_network.lstm_state_out
    
    print(self.thread_index, "In process: len(rewards)", len(self.rewards), "len(states)", len(self.states), "len(actions)", len(self.actions), "len(values)", len(self.values))

    actions = self.actions[:LOCAL_T_MAX]
    states = self.states[:LOCAL_T_MAX]
    rewards = self.rewards[:LOCAL_T_MAX]
    values = self.values[:LOCAL_T_MAX]

    # t_max times loop
    for i in range(len(rewards)):
      self.episode_reward += rewards[i]

      # clip reward
      # rewards[i] = np.clip(rewards[i], -1, 1) ) # TODO: Why do this? Is the range from -1 to 1 problem-specific?

      self.local_t += 1

    if final:
      R = 0.0
    else:
      # print("state", self.states[LOCAL_T_MAX])
      R = self.local_network.run_value(sess, self.states[LOCAL_T_MAX])
    # print(self.thread_index, "initial R", R)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()
    # print("values", values)

    batch_si = []
    batch_ai = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      # print("R", R, "Vi", Vi)
      td = R - Vi
      # a = np.zeros([ACTION_SIZE])
      # a[ai] = 1

      batch_si.append(si)
      batch_ai.append(ai)
      batch_td.append(td)
      batch_R.append(R)

    if final:
      normalized_final_score = self.episode_reward/(self.local_t-self.episode_start_t)
      print("score={}".format(normalized_final_score))
      entropy, action_loss, value_loss, total_loss = self.local_network.run_loss(sess, batch_si[-1], batch_ai[-1], batch_td[-1], batch_R[-1])
      things = {"score": normalized_final_score, 
        "action_loss": action_loss.item(),
        "value_loss": value_loss,
        "entropy": entropy.item(),
        "total_loss": total_loss}
      print("things", things)
      self._record_score(sess, summary_writer, summary_op, summary_inputs,things, global_t) # TODO:NOW: is that "not terminal_end" correct?
      self.episode_start_t = self.local_t
      self.episode_reward = 0

    print(self.thread_index, "Got the following rewards:", rewards, "values", values, "R", R)
    cur_learning_rate = self._anneal_learning_rate(global_t)
    # print(self.thread_index, "Still alive!", cur_learning_rate)

    # print("All the batch stuff", "batch_si", batch_si,
    #               "batch_ai", batch_ai,
    #               "batch_td", batch_td,
    #               "batch_R", batch_R)

    if USE_LSTM:
      batch_si.reverse()
      batch_ai.reverse()
      batch_td.reverse()
      batch_R.reverse()

      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_ai,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.initial_lstm_state: start_lstm_state,
                  self.local_network.step_size : [len(batch_ai)],
                  self.learning_rate_input: cur_learning_rate } )
    else:
      # print("learning_rate_input", cur_learning_rate)
      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_ai,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.learning_rate_input: cur_learning_rate} )
      
    # if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
    if self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL:
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    self.actions = self.actions[LOCAL_T_MAX:]
    self.states = self.states[LOCAL_T_MAX:]
    self.values = self.values[LOCAL_T_MAX:]
    self.rewards = self.rewards[LOCAL_T_MAX:]
    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
    
