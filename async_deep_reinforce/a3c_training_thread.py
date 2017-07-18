# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy as sp
from scipy.stats import norm#, skewnorm
import random
import time
import sys

from game_ac_network import GameACFFNetwork#, GameACLSTMNetwork

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
    self.terminals = []

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

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    # variable controling log output
    self.prev_local_t = 0

    self.acc_state = None

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_mean_values, pi_std_values):
    actions = [norm.rvs(loc=mean, scale=std) for mean, std, skewness in zip(pi_mean_values, pi_std_values)]
    actions[0] = max(0, actions[0]) # Make sure that multiplier isn't negative. There are no negative congestion windows. TODO: Maybe it would be a good idea?
    actions[2] = max(0, actions[2]) # Make sure that the minimum time to wait until you send the next packet isn't negative. 
    return actions

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  def action_step(self, sess, state):
    pi_, value_ = self.local_network.run_policy_and_value(sess, state) # TODO: State is some kind of tuple, isn't it?
    action = self.choose_action(pi_)
    # This whole accumulation thing is just a big hack that is also used in the game implementation. Fortunately with LSTM it's not needed anymore hopefully. 
    if self.acc_state is None:
      self.acc_state = np.stack((state, state, state, state), axis=1)
    else:
      self.acc_state = np.append(self.acc_state[:,1:], state, axis=1)
    states.append(self.acc_state.flatten())
    actions.append(action)
    values.append(value_)
    # if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
    if self.local_t % LOG_INTERVAL == 0:
      print("pi={}".format(pi_))
      print(" V={}".format(value_))
    return action

  def reward_step(self, sess, global_t, summary_writer, summary_op, score_input, reward):
    rewards.append(reward)
    if len(rewards) % LOCAL_T_MAX == 0:
      print("len(rewards)", len(rewards), "len(states)", len(states), "len(actions)", len(actions), "len(values)", len(values))
      return process(sess, global_t, summary_writer, summary_op, score_input)
    # implicitly returns None otherwise

  def final_step(self, sess, global_t, summary_writer, summary_op, score_input, final_state):
    return process(sess, global_t, summary_writer, summary_op, score_input, final_state)

  def process(self, sess, global_t, summary_writer, summary_op, score_input, final_state=None):
    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    # if USE_LSTM:
    #   start_lstm_state = self.local_network.lstm_state_out
    
    # t_max times loop
    for i in range(len(self.states)):
      self.episode_reward += self.rewards[i]

      # clip reward
      # rewards[i] = np.clip(rewards[i], -1, 1) ) # TODO: Why do this? Is the range from -1 to 1 problem-specific?

      self.local_t += 1

    if final_state is not None:
      print("score={}".format(self.episode_reward))
      self._record_score(sess, summary_writer, summary_op, score_input,
                          self.episode_reward, global_t) # TODO:NOW: is that "not terminal_end" correct?
      R = self.local_network.run_value(sess, final_state)
    else:
      R = 0.0

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    cur_learning_rate = self._anneal_learning_rate(global_t)

    if USE_LSTM:
      batch_si.reverse()
      batch_a.reverse()
      batch_td.reverse()
      batch_R.reverse()

      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.initial_lstm_state: start_lstm_state,
                  self.local_network.step_size : [len(batch_a)],
                  self.learning_rate_input: cur_learning_rate } )
    else:
      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
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

    self.states = []
    self.actions = []
    self.rewards = []
    self.values = []
    self.terminals = []

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
    
