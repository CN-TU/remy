# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# tiny = np.finfo(np.float32).tiny
tiny = 1e-20
quite_tiny = 1e-10

from constants import STATE_SIZE
from constants import HIDDEN_SIZE
from constants import ACTION_SIZE
from constants import N_LSTM_LAYERS
import math
import itertools

from tensorflow.python.ops import variable_scope as vs

# # Super cool for debugging reasons! 
# def verbose(original_function):
# 		# make a new function that prints a message when original_function starts and finishes
# 		def new_function(*args, **kwargs):
# 			print('get variable:', '/'.join((tf.get_variable_scope().name, args[0])))
# 			result = original_function(*args, **kwargs)
# 			return result
# 		return new_function
# vs.get_variable = verbose(vs.get_variable)

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
			self.a = tf.placeholder("float", [None, self._action_size], name="a")
		
			# temporary difference (R-V) (input for policy)
			self.td = tf.placeholder("float", [None], name="td")

			# avoid NaN with clipping when value in pi becomes zero
			# log_pi = tf.log(tf.clip_by_value(item, 1e-20, 1.0))
			
			# policy entropy
			# entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

			# entropy = 0.5 * tf.reduce_sum( tf.log(tf.clip_by_value(2*math.pi*self.current_gamma.stddev()**2, 1e-20, float("inf")) ) + 1, axis=1) # TODO: Not sure if minus is required here or not...
			
			self.distribution = tf.contrib.distributions.Gamma(
				concentration=self.pi[1], rate=self.pi[2], allow_nan_stats=False, validate_args=True
			)
			self.distribution_mean = self.distribution.mean()
			self.chosen_action = self.distribution.sample()+self.pi[0]

			self.entropy = entropy_beta * 0.5 * tf.reduce_sum( tf.log(2.0*math.pi*self.distribution.stddev()) + 1, axis=1 ) # TODO: Not sure if minus is required here or not...
			
			# FIXME: axis=1 is stupid because now we only have one action
			# FIXME: Is it a good idea to take the logits of a CDF?
			#                                  this minus isn't necessary...
			# action_loss = tf.reduce_sum( tf.square(tf.subtract( self.pi[0], self.a )), axis=1 )
			# self.action_loss = tf.reduce_sum( tf.square(tf.subtract( self.pi[0], self.a ) / (2.0 * self.pi[1])), axis=1 )
			self.action_loss = tf.reduce_sum( 
				self.distribution.log_prob(tf.clip_by_value(self.a-self.pi[0], tiny, float("inf"))), axis=1 
			) * self.td
			# policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
			# policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ), axis=1 ) * self.td + entropy * entropy_beta )

			self.policy_loss = - tf.reduce_sum( self.action_loss + self.entropy )
			# policy_loss = tf.reduce_sum( action_loss * self.td )

			# R (input for value)
			self.r = tf.placeholder("float", [None], name="r")
			
			# value loss (output)
			# (Learning rate for Critic is half of Actor's, so multiply by 0.5)
			self.value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

			# gradienet of policy and value are summed up
			self.total_loss = self.policy_loss + self.value_loss

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
		weight = tf.Variable(tf.random_uniform(weight_shape))
		bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=lower, maxval=upper))
		return weight, bias

# # Actor-Critic FF Network
# class GameACFFNetwork(GameACNetwork):
# 	def __init__(self,
# 							 thread_index, # -1 for global
# 							 device="/cpu:0"):
# 		GameACNetwork.__init__(self, thread_index, device)

# 		scope_name = "net_" + str(self._thread_index)
# 		with tf.device(self._device), tf.variable_scope(scope_name) as scope:

# 			self.W_state_to_hidden_fc, self.b_state_to_hidden_fc = self._fc_variable([STATE_SIZE, HIDDEN_SIZE])

# 			# print("self.W_state_to_hidden_fc", self.W_state_to_hidden_fc, "self.b_state_to_hidden_fc", self.b_state_to_hidden_fc)

# 			# weight for policy output layer
# 			self.W_hidden_to_action_mean_fc, self.b_hidden_to_action_mean_fc = self._fc_variable([HIDDEN_SIZE, ACTION_SIZE], 100, 1000)
# 			self.W_hidden_to_action_var_fc, self.b_hidden_to_action_var_fc = self._fc_variable([HIDDEN_SIZE, ACTION_SIZE], 1, 10)

# 			# weight for value output layer
# 			self.W_hidden_to_value_fc, self.b_hidden_to_value_fc = self._fc_variable([HIDDEN_SIZE, 1])

# 			# state (input)
# 			self.s = tf.placeholder("float", [None, STATE_SIZE])

# 			# h_fc = tf.nn.relu(tf.matmul(self.s, self.W_state_to_hidden_fc) + self.b_state_to_hidden_fc)
# 			h_fc = tf.matmul(self.s, self.W_state_to_hidden_fc) + self.b_state_to_hidden_fc

# 			raw_pi_loc = tf.matmul(h_fc, self.W_hidden_to_action_mean_fc) + self.b_hidden_to_action_mean_fc
# 			raw_pi_scale = tf.matmul(h_fc, self.W_hidden_to_action_var_fc) + self.b_hidden_to_action_var_fc
# 			# pi_mean = tf.concat([tf.slice(raw_pi_mean,(0,0),(-1,2)), tf.nn.softplus(tf.slice(raw_pi_mean,(0,2),(-1,-1)))], axis=1)
# 			# pi_mean = raw_pi_mean
# 			# policy (output)
# 			# TODO: Now the network is completely linear. And can't map non-linear relationships
# 			self.pi = (
# 				# tf.nn.softplus(raw_pi_loc), # mean
# 				# tf.nn.softplus(raw_pi_scale) #var
# 				tf.nn.softplus(raw_pi_loc), # mean
# 				tf.nn.softplus(raw_pi_scale) #var
# 			)

# 			self.gamma = tf.contrib.distributions.Gamma(self.pi[0], self.pi[1], allow_nan_stats=False, validate_args=True)
# 			self.chosen_action = tf.clip_by_value(self.gamma.sample(), tiny, float("inf"))
# 			# value (output)
# 			v_ = tf.matmul(h_fc, self.W_hidden_to_value_fc) + self.b_hidden_to_value_fc
# 			self.v = tf.reshape( v_, [-1] )

# 	def run_policy_and_value(self, sess, s_t):
# 		pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
# 		# print("run_policy_and_value", pi_out, v_out)
# 		return ((pi_out[0][0], pi_out[1][0]), v_out[0])

# 	def run_policy_action_and_value(self, sess, s_t):
# 		pi_out, action_out, v_out = sess.run( [self.pi, self.chosen_action, self.v], feed_dict = {self.s : [s_t]} )
# 		# print("run_policy_and_value", pi_out, v_out)
# 		return ((pi_out[0][0], pi_out[1][0]), action_out[0], v_out[0])

# 	def run_policy(self, sess, s_t):
# 		pi_out = sess.run( self.pi, feed_dict = {self.s : [s_t]} )
# 		return pi_out[0]

# 	def run_value(self, sess, s_t):
# 		v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
# 		return v_out[0]

# 	def get_vars(self):
# 		return [self.W_state_to_hidden_fc, self.b_state_to_hidden_fc,
# 						self.W_hidden_to_action_mean_fc, self.b_hidden_to_action_mean_fc,
# 						self.W_hidden_to_action_var_fc, self.b_hidden_to_action_var_fc,
# 						self.W_hidden_to_value_fc, self.b_hidden_to_value_fc]

def create_cell(n_hidden):
	return tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden, dropout_keep_prob=1.0)
	# return tf.contrib.rnn.LSTMCell(n_hidden)

all_cell_weight_names = ["/layer_norm_basic_lstm_cell/kernel",
"/layer_norm_basic_lstm_cell/input/gamma",
"/layer_norm_basic_lstm_cell/input/beta",
"/layer_norm_basic_lstm_cell/transform/gamma",
"/layer_norm_basic_lstm_cell/transform/beta",
"/layer_norm_basic_lstm_cell/forget/gamma",
"/layer_norm_basic_lstm_cell/forget/beta",
"/layer_norm_basic_lstm_cell/output/gamma",
"/layer_norm_basic_lstm_cell/output/beta",
"/layer_norm_basic_lstm_cell/state/gamma",
"/layer_norm_basic_lstm_cell/state/beta"]

def lstm_state_tuple(use_np=False):
	if not use_np:
		return tuple([tf.contrib.rnn.LSTMStateTuple(tf.placeholder(tf.float32, [1, HIDDEN_SIZE]),tf.placeholder(tf.float32, [1, HIDDEN_SIZE]))  for _ in range(N_LSTM_LAYERS)])
	else:
		return tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros([1, HIDDEN_SIZE]),np.zeros([1, HIDDEN_SIZE]))  for _ in range(N_LSTM_LAYERS)])

# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
	def __init__(self,
							 thread_index, # -1 for global
							 device="/cpu:0" ):
		GameACNetwork.__init__(self, thread_index, device)

		scope_name = "net_" + str(self._thread_index)
		with tf.device(self._device), tf.variable_scope(scope_name) as scope:

			self.W_state_to_hidden_fc, self.b_state_to_hidden_fc = self._fc_variable([STATE_SIZE, HIDDEN_SIZE])

			# lstm
			# self.lstm = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
			cells = [create_cell(HIDDEN_SIZE) for _ in range(N_LSTM_LAYERS)]
			# print("cells", cells)
			self.lstm = tf.contrib.rnn.MultiRNNCell(
				cells, state_is_tuple=True)

			# weight for policy output layer
			self.W_hidden_to_action_mean_fc, self.b_hidden_to_action_mean_fc = self._fc_variable([HIDDEN_SIZE, ACTION_SIZE],1,2)
			self.W_hidden_to_action_alpha_fc, self.b_hidden_to_action_alpha_fc = self._fc_variable([HIDDEN_SIZE, ACTION_SIZE],10,20)
			self.W_hidden_to_action_beta_fc, self.b_hidden_to_action_beta_fc = self._fc_variable([HIDDEN_SIZE, ACTION_SIZE],1,2)

			# weight for value output layer
			self.W_hidden_to_value_fc, self.b_hidden_to_value_fc = self._fc_variable([HIDDEN_SIZE, 1])

			# state (input)
			self.s = tf.placeholder("float", [None, STATE_SIZE])
		
			h_fc = tf.matmul(self.s, self.W_state_to_hidden_fc) + self.b_state_to_hidden_fc
			# h_fc = tf.matmul(self.s, self.W_state_to_hidden_fc) + self.b_state_to_hidden_fc

			h_fc_reshaped = tf.reshape(h_fc, [1,-1,HIDDEN_SIZE])

			# place holder for LSTM unrolling time step size.
			self.step_size = tf.placeholder(tf.float32, [1])

			# self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, HIDDEN_SIZE])
			# self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, HIDDEN_SIZE])
			# self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
			# 																												self.initial_lstm_state1)
			self.initial_lstm_state = lstm_state_tuple()
			
			# Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
			# When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
			# Unrolling step size is applied via self.step_size placeholder.
			# When forward propagating, step_size is 1.
			# (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
			lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
															h_fc_reshaped,
															initial_state = self.initial_lstm_state,
															sequence_length = self.step_size,
															time_major = False,
															scope = scope)

			# lstm_outputs: (1,5,256) for back prop, (1,1,256) for forward prop.
			# print("lstm_outputs", lstm_outputs)
			
			lstm_outputs = tf.reshape(lstm_outputs, [-1,HIDDEN_SIZE])

			raw_pi_mean = tf.matmul(lstm_outputs, self.W_hidden_to_action_mean_fc) + self.b_hidden_to_action_mean_fc
			raw_pi_alpha = tf.matmul(lstm_outputs, self.W_hidden_to_action_alpha_fc) + self.b_hidden_to_action_alpha_fc
			raw_pi_beta = tf.matmul(lstm_outputs, self.W_hidden_to_action_beta_fc) + self.b_hidden_to_action_beta_fc
			# pi_mean = tf.concat([tf.slice(raw_pi_mean,(0,0),(-1,2)), tf.nn.softplus(tf.slice(raw_pi_mean,(0,2),(-1,-1)))], axis=1)
			# pi_mean = raw_pi_mean
			# policy (output)
			# TODO: Now the network is completely linear. And can't map non-linear relationships
			self.pi = (
				tf.nn.softplus(raw_pi_mean)+1, # mean
				tf.clip_by_value(tf.nn.softplus(raw_pi_alpha), quite_tiny, float("inf")), #alpha
				tf.clip_by_value(tf.nn.softplus(raw_pi_beta), quite_tiny, float("inf")) #beta
			)

			# value (output)
			v_ = tf.matmul(h_fc, self.W_hidden_to_value_fc) + self.b_hidden_to_value_fc
			self.v = tf.reshape( v_, [-1] )

			scope.reuse_variables()

			self.LSTM_variables = [tf.get_variable("multi_rnn_cell/cell_"+str(index)+item) for index, item in itertools.product(range(N_LSTM_LAYERS), all_cell_weight_names)]

			self.reset_state()
			
	def reset_state(self):
		self.lstm_state_out = lstm_state_tuple(use_np=True)

	def run_policy_and_value(self, sess, s_t):
		# This run_policy_and_value() is used when forward propagating.
		# so the step size is 1.
		pi_out, v_out, self.lstm_state_out = sess.run( [self.pi, self.v, self.lstm_state],
														feed_dict = {self.s : [s_t],
														self.initial_lstm_state : self.lstm_state_out,
														# self.initial_lstm_state : self.lstm_state_out[0],
														# self.initial_lstm_state1 : self.lstm_state_out[1],
														self.step_size : [1]} )
		# pi_out: (1,3), v_out: (1)
		return ((pi_out[0][0], pi_out[1][0], pi_out[2][0]), v_out[0])

	def run_policy_action_and_value(self, sess, s_t):
		# This run_policy_and_value() is used when forward propagating.
		# so the step size is 1.
		pi_out, action_out, v_out, self.lstm_state_out = sess.run( [self.pi, self.chosen_action, self.v, self.lstm_state],
														feed_dict = {self.s : [s_t],
														self.initial_lstm_state : self.lstm_state_out,
														# self.initial_lstm_state : self.lstm_state_out[0],
														# self.initial_lstm_state1 : self.lstm_state_out[1],
														self.step_size : [1]} )
		# pi_out: (1,3), v_out: (1)
		return ((pi_out[0][0], pi_out[1][0], pi_out[2][0]), action_out[0], v_out[0])

	def run_policy(self, sess, s_t):
		# This run_policy() is used for displaying the result with display tool.    
		pi_out, self.lstm_state_out = sess.run( [self.pi, self.lstm_state],
													feed_dict = {self.s : [s_t],
													self.initial_lstm_state : self.lstm_state_out,
													#  self.initial_lstm_state0 : self.lstm_state_out[0],
													#  self.initial_lstm_state1 : self.lstm_state_out[1],
													self.step_size : [1]} )

		return (pi_out[0][0], pi_out[1][0])

	def run_value(self, sess, s_t):
		# This run_value() is used for calculating V for bootstrapping at the 
		# end of LOCAL_T_MAX time step sequence.
		# When next sequcen starts, V will be calculated again with the same state using updated network weights,
		# so we don't update LSTM state here.
		prev_lstm_state_out = self.lstm_state_out
		v_out, _ = sess.run( [self.v, self.lstm_state],
								feed_dict = {self.s : [s_t],
								self.initial_lstm_state : self.lstm_state_out,
								# self.initial_lstm_state0 : self.lstm_state_out[0],
								# self.initial_lstm_state1 : self.lstm_state_out[1],
								self.step_size : [1]} )
		
		# roll back lstm state
		self.lstm_state_out = prev_lstm_state_out
		return v_out[0]

	def run_loss(self, sess, si, ai, td, r):
		# This run_value() is used for calculating V for bootstrapping at the 
		# end of LOCAL_T_MAX time step sequence.
		# When next sequcen starts, V will be calculated again with the same state using updated network weights,
		# so we don't update LSTM state here.
		prev_lstm_state_out = self.lstm_state_out
		entropy, action, value, total, window, _ = sess.run( [self.entropy, self.action_loss, self.value_loss, self.total_loss, self.distribution_mean, self.lstm_state],
								feed_dict = {self.s : [si], self.a: [ai], self.td: [td], self.r: [r],
								self.initial_lstm_state : self.lstm_state_out,
								# self.initial_lstm_state0 : self.lstm_state_out[0],
								# self.initial_lstm_state1 : self.lstm_state_out[1],
								self.step_size : [1]} )
		
		# roll back lstm state
		self.lstm_state_out = prev_lstm_state_out
		return entropy, action, value, total, window

	def get_vars(self):
		return [self.W_state_to_hidden_fc, self.b_state_to_hidden_fc,
						self.W_hidden_to_action_mean_fc, self.b_hidden_to_action_mean_fc,
						self.W_hidden_to_action_alpha_fc, self.b_hidden_to_action_alpha_fc,
						self.W_hidden_to_action_beta_fc, self.b_hidden_to_action_beta_fc,
						self.W_hidden_to_value_fc, self.b_hidden_to_value_fc] + self.LSTM_variables
