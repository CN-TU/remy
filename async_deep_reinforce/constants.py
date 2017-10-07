# -*- coding: utf-8 -*-
import tensorflow as tf
import logging
import os

LOCAL_T_MAX = 20 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
# CHECKPOINT_DIR = 'checkpoints'
# LOG_FILE = 'tmp/a3c_log'

# FIXME: Super ugly to hardcode the path!!!
ABSOLUTE_PATH = os.path.join(os.path.expanduser('~'),"repos/remy/")
from os import environ
if environ.get('checkpoints') is not None:
	CHECKPOINT_DIR = ABSOLUTE_PATH+environ.get('checkpoints')
else:
	CHECKPOINT_DIR = ABSOLUTE_PATH+'checkpoints'
LOG_FILE = ABSOLUTE_PATH+'tmp/a3c_log'
ACTOR_FACTOR = 1e0
VALUE_FACTOR = 1e0
GENERAL_FACTOR = 1e-2
INITIAL_ALPHA_LOW = 1e-2*GENERAL_FACTOR   # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e0*GENERAL_FACTOR   # log_uniform high limit for learning rate

PRECISION = tf.float32

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 1e-4
STD_BIAS_OFFSET = -1.5
MAX_TIME_STEP = 1e9
# GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = False # To use GPU, set True
N_LSTM_LAYERS = 3

STATE_SIZE = 11
HIDDEN_SIZE = 256
# ACTION_SIZE = 1 # action size
LAYER_NORMALIZATION = True

DELAY_MULTIPLIER = 0.01
LOG_LEVEL = logging.INFO

SECONDS_NORMALIZER = 1e-2
