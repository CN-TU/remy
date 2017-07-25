#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("Loading a3c...")

import tensorflow as tf
print("Loaded tensorflow")
import numpy as np
print("Loaded numpy")

import signal
import random
import math
import os
import time
import sys

from game_ac_network import GameACFFNetwork#, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import USE_LSTM

print("Loaded a3c!")

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

global_t = 0

stop_requested = False

if USE_LSTM:
  # global_network = GameACLSTMNetwork(ACTION_SIZE, -1, device)
  raise NotImplementedError("LSTM currently doesn't work, what a pity!")
else:
  global_network = GameACFFNetwork(-1, device)

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

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

# print("reported as uninit after init", sess.run(tf.report_uninitialized_variables(tf.global_variables())))

# print("\n\n\nRMSPropApplier", grad_applier, "\n\n\n")

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.summary.scalar("score", score_input)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # set wall time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
    wall_t = float(f.read())
else:
  print("Could not find old checkpoint")
  # set wall time
  wall_t = 0.0

training_threads = {}

# First thread has index 1, 0 is invalid
global_thread_index = 1 # TODO: Is this actually necessary?
idle_threads = set()

def create_training_thread():
  print("Before idle_threads")
  global global_t, global_thread_index, wall_t, sess
  # print("idle_threads", idle_threads)
  if len(idle_threads) == 0:
    print("After idle threads none")
    print("Creating new thread", global_thread_index)
    created_thread = A3CTrainingThread(global_thread_index, global_network, initial_learning_rate,
                                        learning_rate_input,
                                        grad_applier, MAX_TIME_STEP,
                                        device = device)
    training_threads[global_thread_index] = created_thread
    return_index = global_thread_index
    global_thread_index += 1
    # init_new_vars = tf.variables_initializer(created_thread.get_network_vars())
    # sess.run(init_new_vars)
    initialize_uninitialized(sess)
  else:
    print("After idle threads some")
    return_index = idle_threads.pop()
    print("Recycling thread", return_index)
    created_thread = training_threads[return_index]
  # set start time
  start_time = time.time() - wall_t
  created_thread.set_start_time(start_time)

  return return_index

def delete_training_thread(thread_id):
  print("Deleting thread with id", thread_id)
  global sess, global_t, summary_writer, summary_op, score_input
  idle_threads.add(thread_id)
  # del training_threads[thread_id]

def call_process_action(thread_id, state):
  print("call_process_action", thread_id, state)
  global sess, global_t, summary_writer, summary_op, score_input
  chosen_action = tuple(training_threads[thread_id].action_step(sess, state))
  # print("call_process_action", chosen_action)
  return chosen_action

def call_process_reward(thread_id, reward):
  print("call_process_reward", thread_id, reward)
  global sess, global_t, summary_writer, summary_op, score_input
  diff_global_t = training_threads[thread_id].reward_step(sess, global_t, summary_writer, summary_op, score_input, reward)
  if diff_global_t is not None:
    global_t += diff_global_t

def call_process_finished(thread_id, final_state, remove_last):
  print("call_process_finished", thread_id, final_state)
  global sess, global_t, summary_writer, summary_op, score_input
  diff_global_t = training_threads[thread_id].final_step(sess, global_t, summary_writer, summary_op, score_input, final_state, remove_last)
  global_t += diff_global_t

def save_session():
  return
  print("save_session")
  global global_t, sess, CHECKPOINT_DIR
  if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

  # write wall time
  wall_t = time.time() - start_time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'w') as f:
    f.write(str(wall_t))

  saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)

# def train_function(parallel_index):
#   global global_t
  
#   training_thread = training_threads[parallel_index]
#   # set start_time
#   start_time = time.time() - wall_t
#   training_thread.set_start_time(start_time)

#   while True:
#     if stop_requested:
#       break
#     if global_t > MAX_TIME_STEP:
#       break

#     diff_global_t = training_thread.process(sess, global_t, summary_writer,
#                                             summary_op, score_input)
#     global_t += diff_global_t
    
    
# def signal_handler(signal, frame):
#   global stop_requested
#   print('You pressed Ctrl+C!')
#   stop_requested = True
  
# train_threads = []
# for i in range(PARALLEL_SIZE):
#   train_threads.append(threading.Thread(target=train_function, args=(i,)))
  
# signal.signal(signal.SIGINT, signal_handler)

# # set start time
# start_time = time.time() - wall_t

# for t in train_threads:
#   t.start()

# print('Press Ctrl+C to stop')
# signal.pause()

# print('Now saving data. Please wait')
  
# for t in train_threads:
#   t.join()

# if not os.path.exists(CHECKPOINT_DIR):
#   os.mkdir(CHECKPOINT_DIR)

# # write wall time
# wall_t = time.time() - start_time
# wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
# with open(wall_t_fname, 'w') as f:
#   f.write(str(wall_t))

