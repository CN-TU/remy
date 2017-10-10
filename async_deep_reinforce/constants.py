# -*- coding: utf-8 -*-
import tensorflow as tf
import logging
import os
import math

def inverse_softplus(x):
	return math.log(math.exp(x) - 1)

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
ACTOR_FACTOR = 1e-1
VALUE_FACTOR = 1e0
GENERAL_FACTOR = 1e-3
INITIAL_ALPHA_LOW = 1e-2*GENERAL_FACTOR   # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e0*GENERAL_FACTOR   # log_uniform high limit for learning rate

PRECISION = tf.float32

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
ENTROPY_BETA = 1e-4
STD_BIAS_OFFSET = inverse_softplus(0.3)
MAX_TIME_STEP = 1e6
# GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = False # To use GPU, set True
N_LSTM_LAYERS = 3

SECONDS_NORMALIZER = 1e-2

DELAY = 150*SECONDS_NORMALIZER
BIAS_OFFSET = 1
PACKETS_BIAS_OFFSET = inverse_softplus(BIAS_OFFSET)
DELAY_BIAS_OFFSET = inverse_softplus(DELAY)
INTER_PACKET_ARRIVAL_TIME_OFFSET = inverse_softplus(1.0/DELAY)

PACKETS_BIAS_OFFSET = 0
DELAY_BIAS_OFFSET = 0
INTER_PACKET_ARRIVAL_TIME_OFFSET = 0

INITIAL_WINDOW_INCREASE_BIAS_OFFSET = 1e-3
INITIAL_WINDOW_INCREASE_WEIGHT_FACTOR = 1e-4

STATE_SIZE = 11
HIDDEN_SIZE = 256
# ACTION_SIZE = 1 # action size
LAYER_NORMALIZATION = True

LOG_LEVEL = logging.INFO
