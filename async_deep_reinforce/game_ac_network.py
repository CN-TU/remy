# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from constants import STATE_SIZE
from constants import HIDDEN_SIZE
from constants import ACTION_SIZE
import math

# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
  def __init__(self,
               thread_index, # -1 for global
               device="/cpu:0"):
    self._action_size = ACTION_SIZE
    self._thread_index = thread_index
    self._device = device

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      # log_pi = tf.log(tf.clip_by_value(item, 1e-20, 1.0))
      
      # policy entropy
      # entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

      # entropy = 0.5 * tf.reduce_sum( tf.log(tf.clip_by_value(2*math.pi*self.current_gamma.stddev()**2, 1e-20, float("inf")) ) + 1, axis=1) # TODO: Not sure if minus is required here or not...
      
      entropy = 0.5 * tf.reduce_sum( tf.log(2*math.pi*self.gamma.variance()) + 1, axis=1) # TODO: Not sure if minus is required here or not...
      
      # FIXME: axis=1 is stupid because now we only have one action
      # FIXME: Is it a good idea to take the logits of a CDF?
      #                                  this minus isn't necessary...
      action_loss = tf.reduce_sum( tf.square(tf.subtract( self.gamma.mean(), self.a )), axis=1 )
      # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
      # policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ), axis=1 ) * self.td + entropy * entropy_beta )

      policy_loss = tf.reduce_sum( action_loss * self.td - entropy * entropy_beta )
      # policy_loss = tf.reduce_sum( action_loss * self.td )

      # R (input for value)
      self.r = tf.placeholder("float", [None])
      
      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss

  def run_policy_and_value(self, sess, s_t):
    raise NotImplementedError()
    
  def run_policy(self, sess, s_t):
    raise NotImplementedError()

  def run_value(self, sess, s_t):
    raise NotImplementedError()    

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_network, name=None):
    src_vars = src_network.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "GameACNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_variable(self, weight_shape, lower=None, upper=None):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    if lower is None:
      lower = -d
    if upper is None:
      upper = d
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=lower, maxval=upper))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=lower, maxval=upper))
    return weight, bias

  # def _conv_variable(self, weight_shape):
  #   w = weight_shape[0]
  #   h = weight_shape[1]
  #   input_channels  = weight_shape[2]
  #   output_channels = weight_shape[3]
  #   d = 1.0 / np.sqrt(input_channels * w * h)
  #   bias_shape = [output_channels]
  #   weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
  #   bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
  #   return weight, bias

  # def _conv2d(self, x, W, stride):
  #   return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
  def __init__(self,
               thread_index, # -1 for global
               device="/cpu:0"):
    GameACNetwork.__init__(self, thread_index, device)

    scope_name = "net_" + str(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:

      self.W_state_to_hidden_fc, self.b_state_to_hidden_fc = self._fc_variable([STATE_SIZE, HIDDEN_SIZE])

      # print("self.W_state_to_hidden_fc", self.W_state_to_hidden_fc, "self.b_state_to_hidden_fc", self.b_state_to_hidden_fc)

      # weight for policy output layer
      self.W_hidden_to_action_mean_fc, self.b_hidden_to_action_mean_fc = self._fc_variable([HIDDEN_SIZE, ACTION_SIZE], 100, 1000)
      self.W_hidden_to_action_var_fc, self.b_hidden_to_action_var_fc = self._fc_variable([HIDDEN_SIZE, ACTION_SIZE], 1, 10)

      # weight for value output layer
      self.W_hidden_to_value_fc, self.b_hidden_to_value_fc = self._fc_variable([HIDDEN_SIZE, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, STATE_SIZE])

      h_fc = tf.nn.relu(tf.matmul(self.s, self.W_state_to_hidden_fc) + self.b_state_to_hidden_fc)

      raw_pi_loc = tf.matmul(h_fc, self.W_hidden_to_action_mean_fc) + self.b_hidden_to_action_mean_fc
      raw_pi_scale = tf.matmul(h_fc, self.W_hidden_to_action_var_fc) + self.b_hidden_to_action_var_fc
      # pi_mean = tf.concat([tf.slice(raw_pi_mean,(0,0),(-1,2)), tf.nn.softplus(tf.slice(raw_pi_mean,(0,2),(-1,-1)))], axis=1)
      # pi_mean = raw_pi_mean
      # policy (output)
      # TODO: Now the network is completely linear. And can't map non-linear relationships
      self.pi = (
        # tf.nn.softplus(raw_pi_loc), # mean
        # tf.nn.softplus(raw_pi_scale) #var
        tf.nn.softplus(raw_pi_loc), # mean
        tf.nn.softplus(raw_pi_scale) #var
      )

      self.gamma = tf.contrib.distributions.Gamma(self.pi[0], self.pi[1], allow_nan_stats=False, validate_args=True)
      self.chosen_action = tf.clip_by_value(self.gamma.sample(), 1e-20, float("inf"))
      # value (output)
      v_ = tf.matmul(h_fc, self.W_hidden_to_value_fc) + self.b_hidden_to_value_fc
      self.v = tf.reshape( v_, [-1] )

  def run_policy_and_value(self, sess, s_t):
    pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
    # print("run_policy_and_value", pi_out, v_out)
    return ((pi_out[0][0], pi_out[1][0]), v_out[0])

  def run_policy_action_and_value(self, sess, s_t):
    pi_out, action_out, v_out = sess.run( [self.pi, self.chosen_action, self.v], feed_dict = {self.s : [s_t]} )
    # print("run_policy_and_value", pi_out, v_out)
    return ((pi_out[0][0], pi_out[1][0]), action_out[0], v_out[0])

  def run_policy(self, sess, s_t):
    pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
    return pi_out[0]

  def run_value(self, sess, s_t):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0]

  def get_vars(self):
    return [self.W_state_to_hidden_fc, self.b_state_to_hidden_fc,
            self.W_hidden_to_action_mean_fc, self.b_hidden_to_action_mean_fc,
            self.W_hidden_to_action_var_fc, self.b_hidden_to_action_var_fc,
            self.W_hidden_to_value_fc, self.b_hidden_to_value_fc]

# # Actor-Critic LSTM Network
# class GameACLSTMNetwork(GameACNetwork):
#   def __init__(self,
#                action_size,
#                thread_index, # -1 for global
#                device="/cpu:0" ):
#     GameACNetwork.__init__(self, action_size, thread_index, device)

#     scope_name = "net_" + str(self._thread_index)
#     with tf.device(self._device), tf.variable_scope(scope_name) as scope:
#       self.W_conv1, self.b_conv1 = self._conv_variable([8, 8, 4, 16])  # stride=4,
#       self.W_conv2, self.b_conv2 = self._conv_variable([4, 4, 16, 32]) # stride=2
      
#       self.W_state_to_hidden_fc, self.b_state_to_hidden_fc = self._fc_variable([2592, 256])

#       # lstm
#       self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

#       # weight for policy output layer
#       self.W_hidden_to_action_fc, self.b_hidden_to_action_fc = self._fc_variable([256, action_size])

#       # weight for value output layer
#       self.W_hidden_to_value_fc, self.b_hidden_to_value_fc = self._fc_variable([256, 1])

#       # state (input)
#       self.s = tf.placeholder("float", [None, 84, 84, 4])
    
#       h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 4) + self.b_conv1)
#       h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)

#       h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
#       h_state_to_hidden_fc = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_state_to_hidden_fc) + self.b_state_to_hidden_fc)
#       # h_state_to_hidden_fc shape=(5,256)

#       h_state_to_hidden_fc_reshaped = tf.reshape(h_state_to_hidden_fc, [1,-1,256])
#       # h_fc_reshaped = (1,5,256)

#       # place holder for LSTM unrolling time step size.
#       self.step_size = tf.placeholder(tf.float32, [1])

#       self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
#       self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
#       self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
#                                                               self.initial_lstm_state1)
      
#       # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
#       # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
#       # Unrolling step size is applied via self.step_size placeholder.
#       # When forward propagating, step_size is 1.
#       # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
#       lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
#                                                         h_state_to_hidden_fc_reshaped,
#                                                         initial_state = self.initial_lstm_state,
#                                                         sequence_length = self.step_size,
#                                                         time_major = False,
#                                                         scope = scope)

#       # lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.
      
#       lstm_outputs = tf.reshape(lstm_outputs, [-1,256])

#       # policy (output)
#       self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_hidden_to_action_fc) + self.b_hidden_to_action_fc)
      
#       # value (output)
#       v_ = tf.matmul(lstm_outputs, self.W_hidden_to_value_fc) + self.b_hidden_to_value_fc
#       self.v = tf.reshape( v_, [-1] )

#       scope.reuse_variables()
#       self.W_lstm = tf.get_variable("basic_lstm_cell/weights")
#       self.b_lstm = tf.get_variable("basic_lstm_cell/biases")

#       self.reset_state()
      
#   def reset_state(self):
#     self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
#                                                         np.zeros([1, 256]))

#   def run_policy_and_value(self, sess, s_t):
#     # This run_policy_and_value() is used when forward propagating.
#     # so the step size is 1.
#     pi_out, v_out, self.lstm_state_out = sess.run( [self.pi, self.v, self.lstm_state],
#                                                    feed_dict = {self.s : [s_t],
#                                                                 self.initial_lstm_state0 : self.lstm_state_out[0],
#                                                                 self.initial_lstm_state1 : self.lstm_state_out[1],
#                                                                 self.step_size : [1]} )
#     # pi_out: (1,3), v_out: (1)
#     return (pi_out[0], v_out[0])

#   def run_policy(self, sess, s_t):
#     # This run_policy() is used for displaying the result with display tool.    
#     pi_out, self.lstm_state_out = sess.run( [self.pi, self.lstm_state],
#                                             feed_dict = {self.s : [s_t],
#                                                          self.initial_lstm_state0 : self.lstm_state_out[0],
#                                                          self.initial_lstm_state1 : self.lstm_state_out[1],
#                                                          self.step_size : [1]} )
                                            
#     return pi_out[0]

#   def run_value(self, sess, s_t):
#     # This run_value() is used for calculating V for bootstrapping at the 
#     # end of LOCAL_T_MAX time step sequence.
#     # When next sequcen starts, V will be calculated again with the same state using updated network weights,
#     # so we don't update LSTM state here.
#     prev_lstm_state_out = self.lstm_state_out
#     v_out, _ = sess.run( [self.v, self.lstm_state],
#                          feed_dict = {self.s : [s_t],
#                                       self.initial_lstm_state0 : self.lstm_state_out[0],
#                                       self.initial_lstm_state1 : self.lstm_state_out[1],
#                                       self.step_size : [1]} )
    
#     # roll back lstm state
#     self.lstm_state_out = prev_lstm_state_out
#     return v_out[0]

#   # def get_vars(self):
#   #   return [self.W_conv1, self.b_conv1,
#   #           self.W_conv2, self.b_conv2,
#   #           self.W_state_to_hidden_fc, self.b_state_to_hidden_fc,
#   #           self.W_lstm, self.b_lstm,
#   #           self.W_hidden_to_action_fc, self.b_hidden_to_action_fc,
#   #           self.W_hidden_to_value_fc, self.b_hidden_to_value_fc]
