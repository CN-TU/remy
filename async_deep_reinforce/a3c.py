#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
cooperative = "cooperative" in sys.argv
print("Cooperative:", cooperative)

import datetime

import tensorflow as tf
import numpy as np

import signal
import random
import math
import os
import time

from game_ac_network import GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
# from rmsprop_applier import RMSPropApplier

# from constants import ACTION_SIZE
# from constants import INITIAL_ALPHA_LOW
# from constants import INITIAL_ALPHA_HIGH
# from constants import INITIAL_ALPHA_LOG_RATE
from constants import INITIAL_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import USE_GPU
from constants import PRECISION
from constants import LOG_LEVEL

from os import environ

import logging
logging.basicConfig(level=LOG_LEVEL)

device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

def initialize_uninitialized(sess):
  with tf.device(device):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    logging.info("initializing uninitialized variables")
    logging.debug(" ".join(map(str,("initializing uninitialized variables:", [str(i.name) for i in not_initialized_vars]))))  # only for testing
    if len(not_initialized_vars) > 0:
        sess.run(tf.variables_initializer(not_initialized_vars))

# initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
#                                     INITIAL_ALPHA_HIGH,
#                                     INITIAL_ALPHA_LOG_RATE)

initial_learning_rate = INITIAL_RATE

global_t = 0

if cooperative:
  global_network = GameACLSTMNetwork(0, device)
else:
  num_threads = int(environ.get('num_threads'))
  logging.info(" ".join(map(str,("num_threads", num_threads))))

  assert(num_threads is not None)
  global_network = []
  for i in range(1, int(num_threads)+1):
    global_network.append(GameACLSTMNetwork(-i, device))

learning_rate_input = tf.placeholder(PRECISION)

# grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
#                               decay = RMSP_ALPHA,
#                               momentum = 0.0,
#                               epsilon = RMSP_EPSILON,
#                               clip_norm = GRAD_NORM_CLIP,
#                               device = device)

with tf.device(device):
  grad_applier = tf.train.RMSPropOptimizer(
    learning_rate = learning_rate_input,
    decay = RMSP_ALPHA,
    momentum = 0.0,
    epsilon = RMSP_EPSILON
  )

# grad_applier = tf.train.GradientDescentOptimizer(learning_rate = learning_rate_input)

# for i in range(PARALLEL_SIZE):
#   training_thread = A3CTrainingThread(i, global_network, initial_learning_rate,
#                                       learning_rate_input,
#                                       grad_applier, MAX_TIME_STEP,
#                                       device = device)
#   training_threads.append(training_thread)

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

# init or load checkpoint with saver
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

saver = tf.train.Saver()
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  logging.info(" ".join(map(str,("checkpoint loaded:", checkpoint.model_checkpoint_path))))
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[-1])
  logging.info(" ".join(map(str,(">>> global step set:", global_t))))
  # set wall time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
    wall_t = float(f.read())
  current_datetime_fname = CHECKPOINT_DIR + '/' + 'current_datetime.' + str(global_t)
  with open(current_datetime_fname, 'r') as f:
    current_datetime = f.read()
else:
  logging.info("Could not find old checkpoint")
  # set wall time
  wall_t = 0.0
  current_datetime = datetime.datetime.now().isoformat()[:-7]

# summary for tensorboard
with tf.device(device):
  score_throughput = tf.placeholder(PRECISION)
  score_delay = tf.placeholder(PRECISION)
  entropy = tf.placeholder(PRECISION)
  actor_loss = tf.placeholder(PRECISION)
  value_loss = tf.placeholder(PRECISION)
  total_loss = tf.placeholder(PRECISION)
  window = tf.placeholder(PRECISION)
  window_increase = tf.placeholder(PRECISION)
  std = tf.placeholder(PRECISION)
  # speed = tf.placeholder(PRECISION)
  tf.summary.scalar("score_throughput", score_throughput)
  tf.summary.scalar("score_delay", score_delay)
  tf.summary.scalar("entropy", entropy)
  tf.summary.scalar("actor_loss", actor_loss)
  tf.summary.scalar("value_loss", value_loss)
  tf.summary.scalar("total_loss", total_loss)
  tf.summary.scalar("window", window)
  tf.summary.scalar("window_increase", window_increase)
  tf.summary.scalar("std", std)
  # tf.summary.scalar("speed", speed)
  summary_inputs = {
    "score_throughput": score_throughput,
    "score_delay": score_delay,
    "entropy": entropy,
    "actor_loss": actor_loss,
    "value_loss": value_loss,
    "total_loss": total_loss,
    "window": window,
    "window_increase": window_increase,
    "std": std,
    # "speed": speed
  }

  summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(LOG_FILE+'/'+current_datetime, sess.graph)

training_threads = {}

# First thread has index 1, 0 is invalid
global_thread_index = 1
idle_threads = set()

# start_time = time.time() - wall_t

global training

def create_training_thread(training, delay_delta):
  # print("Oh yeah, creating thraining thread!!!")
  logging.info(" ".join(map(str,("training", training))))
  global global_t, global_thread_index, wall_t, sess
  if len(idle_threads) == 0:
    logging.info(" ".join(map(str,("Creating new thread", global_thread_index))))
    if cooperative:
      created_thread = A3CTrainingThread(global_thread_index, global_network, initial_learning_rate,
                                          learning_rate_input,
                                          grad_applier, MAX_TIME_STEP,
                                          device, training, cooperative,
                                          delay_delta)
    else:
      created_thread = A3CTrainingThread(global_thread_index, global_network[-global_thread_index], initial_learning_rate,
                                          learning_rate_input,
                                          grad_applier, MAX_TIME_STEP,
                                          device, training, cooperative,
                                          delay_delta)
    training_threads[global_thread_index] = created_thread
    return_index = global_thread_index
    global_thread_index += 1
    initialize_uninitialized(sess)
  else:
    return_index = idle_threads.pop()
    logging.info(" ".join(map(str,("Recycling thread", return_index))))
    created_thread = training_threads[return_index]

  # set start time
  # start_time = time.time() - wall_t
  # created_thread.set_start_time(start_time)
  created_thread.episode_count = 0

  created_thread.time_differences = []
  created_thread.windows = []
  created_thread.states = []
  created_thread.actions = []
  created_thread.ticknos = []
  created_thread.rewards = []
  created_thread.durations = []
  created_thread.values = []
  created_thread.estimated_values = []
  created_thread.start_lstm_states = []
  created_thread.variable_snapshots = []
  created_thread.local_t = 0
  created_thread.episode_reward_throughput = 0
  created_thread.episode_reward_delay = 0
  created_thread.local_network.reset_state()
  sess.run( created_thread.sync )

  return return_index

def delete_training_thread(thread_id):
  # logging.info(" ".join(map(str,("Deleting thread with id", thread_id))))
  global sess, global_t, summary_writer, summary_op, summary_inputs
  idle_threads.add(thread_id)
  # del training_threads[thread_id]

def call_process_action(thread_id, state, tickno, window):
  # print("state", state)
  # logging.debug(" ".join(map(str,("call_process_action", thread_id, state, tickno))))
  global sess, global_t, summary_writer, summary_op, summary_inputs
  chosen_action = training_threads[thread_id].action_step(sess, state, tickno, window)
  return chosen_action

def call_process_reward(thread_id, reward_throughput, reward_delay, duration):
  # logging.debug(" ".join(map(str,("call_process_reward", thread_id, reward_throughput, reward_delay, duration))))
  global sess, global_t, summary_writer, summary_op, summary_inputs
  diff_global_t = training_threads[thread_id].reward_step(sess, global_t, summary_writer, summary_op, summary_inputs, reward_throughput, reward_delay, duration)
  global_t += diff_global_t
  if global_t >= MAX_TIME_STEP:
    logging.info("Reached maximum time step, saving and terminating...")
    save_session()
    sys.exit()

def call_process_finished(thread_id, actions_to_remove, time_difference, window):
  # logging.debug(" ".join(map(str,("call_process_finished", thread_id, time_difference))))
  global sess, global_t, summary_writer, summary_op, summary_inputs
  training_threads[thread_id].final_step(sess, global_t, summary_writer, summary_op, summary_inputs, actions_to_remove, time_difference, window)

def save_session():
  # logging.debug("save_session")
  global global_t, sess, CHECKPOINT_DIR
  if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

  # write wall time
  # wall_t = time.time() - start_time
  # FIXME: Currently completely useless...
  wall_t = time.time()
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'w') as f:
    f.write(str(wall_t))

  current_datetime_fname = CHECKPOINT_DIR + '/' + 'current_datetime.' + str(global_t)
  with open(current_datetime_fname, 'w') as f:
    f.write(str(current_datetime))

  saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint' + "-" + current_datetime, global_step = global_t)
