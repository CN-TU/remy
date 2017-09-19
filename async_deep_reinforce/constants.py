# -*- coding: utf-8 -*-
import tensorflow as tf
import logging

LOCAL_T_MAX = 20 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
# CHECKPOINT_DIR = 'checkpoints'
# LOG_FILE = 'tmp/a3c_log'

# FIXME: Super ugly to hardcode the path!!!
ABSOLUTE_PATH = "~/repos/remy/"
CHECKPOINT_DIR = ABSOLUTE_PATH+'checkpoints'
LOG_FILE = ABSOLUTE_PATH+'tmp/a3c_log'
INITIAL_ALPHA_LOW = 1e-4    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-2   # log_uniform high limit for learning rate

PRECISION = tf.float64

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 1e-4
MAX_TIME_STEP = 10 * 1e7
# GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = False # To use GPU, set True
N_LSTM_LAYERS = 3

STATE_SIZE = 11
HIDDEN_SIZE = 128
# ACTION_SIZE = 1 # action size
LAYER_NORMALIZATION = True

# Why? Because the guys of Remy also use it...
DELAY_MULTIPLIER = 1.0
LOG_LEVEL = logging.INFO
# MINIMUM_STD = 0.5
MINIMUM_STD = 1e-2
ACTOR_FACTOR = 1.0
