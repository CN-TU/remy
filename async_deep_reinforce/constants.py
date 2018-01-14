# -*- coding: utf-8 -*-
import tensorflow as tf
import logging
import os
import math
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL)

tiny = 1e-10

def inverse_softplus(x):
	return math.log(math.exp(x) - 1)

LOCAL_T_MAX = 20 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
# CHECKPOINT_DIR = 'checkpoints'
# LOG_FILE = 'tmp/a3c_log'

# FIXME: Super ugly to hardcode the path!!!
# ABSOLUTE_PATH = os.path.join(os.path.expanduser('~'),"repos/remy/")
ABSOLUTE_PATH = os.path.join(os.getcwd())
if os.environ.get('checkpoints') is not None:
	CHECKPOINT_DIR = os.path.join(ABSOLUTE_PATH,os.environ.get('checkpoints'))
else:
	CHECKPOINT_DIR = os.path.join(ABSOLUTE_PATH,'checkpoints')
if not os.path.exists(CHECKPOINT_DIR):
	os.makedirs(CHECKPOINT_DIR)

logging.info(" ".join(map(str,("CHECKPOINT_DIR:",CHECKPOINT_DIR))))

LOG_FILE = os.path.join(ABSOLUTE_PATH,'a3c_log')
logging.info(" ".join(map(str,("LOG_DIR:",LOG_FILE))))
ACTOR_FACTOR = 1e0
VALUE_FACTOR = 1e0
GENERAL_FACTOR = 10**-2
# INITIAL_ALPHA_LOW = 1e-2*GENERAL_FACTOR   # log_uniform low limit for learning rate
# INITIAL_ALPHA_HIGH = 1e0*GENERAL_FACTOR   # log_uniform high limit for learning rate
INITIAL_RATE = GENERAL_FACTOR

PRECISION = tf.float32

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
ENTROPY_BETA = 1e-4
# STD_BIAS_OFFSET = 0.3
STD_BIAS_OFFSET = inverse_softplus(0.5)
STD_BIAS_MINIMUM = 0.0
# STD_BIAS_OFFSET = inverse_softplus(0.1)
# STD_BIAS_OFFSET = 0
MAX_TIME_STEP = 1e8
# GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = False # To use GPU, set True
N_LSTM_LAYERS = int(os.environ.get('layers')) if os.environ.get('layers') is not None else 2

assert(os.environ.get('rtt') is not None)
DELAY = float(os.environ.get('rtt'))
BIAS_OFFSET = 1
# PACKETS_BIAS_OFFSET = inverse_softplus(BIAS_OFFSET)
# DELAY_BIAS_OFFSET = inverse_softplus(DELAY)
# INTER_PACKET_ARRIVAL_TIME_OFFSET = inverse_softplus(1.0/DELAY)
# SENT_OFFSET = inverse_softplus(BIAS_OFFSET)

PACKETS_BIAS_OFFSET = inverse_softplus(BIAS_OFFSET)
# DELAY_BIAS_OFFSET = inverse_softplus(DELAY)
INTER_PACKET_ARRIVAL_TIME_OFFSET = inverse_softplus(BIAS_OFFSET)
SENT_OFFSET = inverse_softplus(BIAS_OFFSET)

PACKETS_BIAS_OFFSET = BIAS_OFFSET
# DELAY_BIAS_OFFSET = inverse_softplus(DELAY)
INTER_PACKET_ARRIVAL_TIME_OFFSET = 1/DELAY
SENT_OFFSET = BIAS_OFFSET

# PACKETS_BIAS_OFFSET = inverse_softplus(BIAS_OFFSET/(1-GAMMA))
# DELAY_BIAS_OFFSET = inverse_softplus(DELAY/(1-GAMMA))
# INTER_PACKET_ARRIVAL_TIME_OFFSET = inverse_softplus(DELAY/(1-GAMMA))
# LOST_OFFSET = inverse_softplus(1e-10/(1-GAMMA))

INITIAL_WINDOW_INCREASE_BIAS_OFFSET = 0
INITIAL_WINDOW_INCREASE_WEIGHT_FACTOR = 1e-2
PACKET_FACTOR = 1e-2
DURATION_FACTOR = 1e-2

# INITIAL_WINDOW_INCREASE_BIAS_OFFSET = 0
# INITIAL_WINDOW_INCREASE_WEIGHT_FACTOR = 1
# PACKET_FACTOR = 1
# DURATION_FACTOR = 1

STATE_SIZE = int(os.environ.get('state_size')) if os.environ.get('state_size') is not None else 15
HIDDEN_SIZE = int(os.environ.get('hidden_size')) if os.environ.get('hidden_size') is not None else 32
# ACTION_SIZE = 1 # action size

SIGMOID_ALPHA = 10.0

MIN_WINDOW = 10.0
MAX_WINDOW = 1000.0
