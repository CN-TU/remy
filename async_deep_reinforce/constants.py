# -*- coding: utf-8 -*-

LOCAL_T_MAX = 20 # repeat step size
RMSP_ALPHA = 0.99 # decay parameter for RMSProp
RMSP_EPSILON = 0.1 # epsilon parameter for RMSProp
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'tmp/a3c_log'
INITIAL_ALPHA_LOW = 1e-5    # log_uniform low limit for learning rate
INITIAL_ALPHA_HIGH = 1e-3   # log_uniform high limit for learning rate

PARALLEL_SIZE = 8 # parallel thread size
ROM = "pong.bin"     # action size = 3

INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
GAMMA = 0.99 # discount factor for rewards
# ENTROPY_BETA = 0.01 # entropy regurarlization constant (discrete actions)
# ENTROPY_BETA = 10**(-4) # entropy regurarlization constant (continuous actions)
ENTROPY_BETA = 10**(-4) # entropy regurarlization constant (continuous actions)
MAX_TIME_STEP = 10 * 10**7
GRAD_NORM_CLIP = 20.0 # gradient norm clipping
USE_GPU = False # To use GPU, set True
# USE_LSTM = True # True for A3C LSTM, False for A3C FF
USE_LSTM = True
N_LSTM_LAYERS = 3

# Max's stuff
# STATE_SIZE = 4
STATE_SIZE = 7 # Using his weird hack of using the last 4 states.
HIDDEN_SIZE = 128
ACTION_SIZE = 1 # action size
