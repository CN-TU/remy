# -*- coding: utf-8 -*-
import tensorflow as tf
ds = tf.contrib.distributions
import numpy as np

from constants import STATE_SIZE
from constants import HIDDEN_SIZE
from constants import N_LSTM_LAYERS
from constants import PRECISION
from constants import LAYER_NORMALIZATION
from constants import ENTROPY_BETA
from constants import MINIMUM_STD
from constants import ACTOR_FACTOR

# tiny = np.finfo(np.float32).tiny
tiny = 1e-20
log_smallest = np.log(np.finfo(np.float32).tiny) if PRECISION==tf.float32 else np.log(np.finfo(np.float64).tiny)
quite_tiny = 1e-10
not_that_tiny = 1e-5
not_tiny_at_all = 1e-1
one = 1

import math
import numpy as np
import numpy.random
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
		# self._action_size = ACTION_SIZE
		self._thread_index = thread_index
		self._device = device

	def prepare_loss(self):
		with tf.device(self._device):
			# taken action (input for policy)
			self.a = tf.placeholder(PRECISION, [None, 1], name="a")

			# temporary difference (R-V) (input for policy)
			self.td_throughput = tf.placeholder(PRECISION, [None], name="td_throughput")
			self.td_delay = tf.placeholder(PRECISION, [None], name="td_delay")

			normal_variance = tf.log((self.pi[1]*self.pi[1])/(self.pi[0]*self.pi[0])+1.0)
			self.distribution = ds.TransformedDistribution(
					distribution=ds.Normal(loc=tf.log(self.pi[0])-normal_variance/2.0, scale=tf.sqrt(normal_variance), allow_nan_stats=False, validate_args=True),
					bijector=ds.bijectors.Exp(),
					name="LogNormalTransformedDistribution")

			self.distribution_mean = self.pi[0]
			self.distribution_std = self.pi[1]
			self.inner_distribution_mean = self.distribution.distribution.mean()
			self.inner_distribution_std = self.distribution.distribution.stddev()
			self.chosen_action = tf.ceil(self.distribution.sample())
			# self.chosen_action = self.distribution.sample() + OFFSET

			# policy entropy
			self.entropy = ENTROPY_BETA * tf.reduce_sum(self.distribution.distribution.mean() + 0.5 * tf.log(2.0*math.pi*math.e*self.distribution.distribution.variance()), axis=1)
			# self.entropy = ENTROPY_BETA * tf.reduce_sum(0.5 * tf.log(2.0*math.pi*math.e*self.pi[1]*self.pi[1]), axis=1)
			self.skewness = ENTROPY_BETA * tf.reduce_sum((tf.exp(self.distribution.distribution.variance()) + 2.0)*tf.sqrt(tf.exp(self.distribution.distribution.variance()) - 1.0), axis=1)
			# self.actor_loss = tf.reduce_sum(self.distribution.log_prob(self.a - OFFSET), axis=1) * (self.td_throughput + self.td_delay)
			self.actor_loss = tf.reduce_sum(tf.log(self.distribution.cdf(self.a) - self.distribution.cdf(tf.clip_by_value(self.a - 1.0, tiny, float("inf")))), axis=1) * (self.td_throughput + self.td_delay)

			# self.policy_loss = - tf.reduce_sum(self.actor_loss + ENTROPY_BETA * self.entropy - SKEWNESS_GAMMA * self.skewness)
			# self.policy_loss = - ACTOR_FACTOR * tf.reduce_sum(self.actor_loss + self.entropy)
			self.policy_loss = - ACTOR_FACTOR * tf.reduce_sum(self.actor_loss)

			# R (input for value)
			self.r_packets = tf.placeholder(PRECISION, [None], name="r_packets")
			self.r_accumulated_delay = tf.placeholder(PRECISION, [None], name="r_accumulated_delay")
			self.r_duration = tf.placeholder(PRECISION, [None], name="r_duration")

			# value loss (output)
			# (Learning rate for Critic is half of Actor's, so multiply by 0.5)
			# TODO: Why is the learning rate half of the Actor's? Sources?
			# self.value_loss = 0.5 * (tf.nn.l2_loss(self.r_packets - self.v_packets) + tf.nn.l2_loss(self.r_accumulated_delay - self.v_accumulated_delay) + tf.nn.l2_loss(self.r_duration - self.v_duration))
			self.value_loss = (tf.nn.l2_loss(self.r_packets - self.v_packets) + tf.nn.l2_loss(self.r_accumulated_delay - self.v_accumulated_delay) + tf.nn.l2_loss(self.r_duration - self.v_duration))

			# gradient of policy and value are summed up
			self.total_loss = self.policy_loss + self.value_loss

	def run_policy_and_value(self, sess, s_t):
		raise NotImplementedError()

	def run_policy(self, sess, s_t):
		raise NotImplementedError()

	def run_value(self, sess, s_t):
		raise NotImplementedError()

	def get_vars(self):
		raise NotImplementedError()

	def backup_vars(self, name=None):
		def backup_vars_inner_function():
			self.backup_lstm_state = self.lstm_state
		return backup_vars_inner_function

	def restore_backup(self, name=None):
		def restore_backup_inner_function():
			self.lstm_state = self.backup_lstm_state
		return restore_backup_inner_function

	# def backup_vars(self, name=None):
	# 	self.backup_vars = list(map(lambda var: tf.Variable(tf.zeros(var.get_shape(), dtype=PRECISION)), self.get_vars()))
	# 	sync_ops = []

	# 	with tf.name_scope(name, "GameACNetwork", []) as name:
	# 		with tf.device(self._device):
	# 			for (src_var, dst_var) in zip(self.get_vars(), self.backup_vars):
	# 				sync_op = tf.assign(dst_var, src_var)
	# 				sync_ops.append(sync_op)

	# 			group = tf.group(*sync_ops, name=name)
	# 			def backup_vars_inner_function(sess):
	# 				self.backup_lstm_state = self.lstm_state
	# 				sess.run(group)

	# 			return backup_vars_inner_function

	# def restore_backup(self, name=None):
	# 	sync_ops = []

	# 	with tf.name_scope(name, "GameACNetwork", []) as name:
	# 		with tf.device(self._device):
	# 			for (src_var, dst_var) in zip(self.backup_vars, self.get_vars()):
	# 				sync_op = tf.assign(dst_var, src_var)
	# 				sync_ops.append(sync_op)

	# 			group = tf.group(*sync_ops, name=name)
	# 			def restore_backup_inner_function(sess):
	# 				self.lstm_state = self.backup_lstm_state
	# 				sess.run(group)

	# 			return restore_backup_inner_function

	def sync_from(self, src_network, name=None):
		src_vars = src_network.get_vars()
		dst_vars = self.get_vars()

		sync_ops = []

		with tf.device(self._device):
			with tf.name_scope(name, "GameACNetwork", []) as name:
				for (src_var, dst_var) in zip(src_vars, dst_vars):
					sync_op = tf.assign(dst_var, src_var)
					sync_ops.append(sync_op)

				return tf.group(*sync_ops, name=name)

	# weight initialization based on muupan's code
	# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
	def _fc_variable(self, weight_shape, factor=1.0):
		input_channels  = weight_shape[0]
		output_channels = weight_shape[1]
		d = 1.0 / np.sqrt(input_channels)
		d *= factor
		bias_shape = [output_channels]
		weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d, dtype=PRECISION))
		bias   = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d, dtype=PRECISION))
		return weight, bias

def lstm_state_tuple(use_np=False):
	if not use_np:
		return tuple([tf.contrib.rnn.LSTMStateTuple(tf.placeholder(PRECISION, [1, HIDDEN_SIZE]),tf.placeholder(PRECISION, [1, HIDDEN_SIZE]))  for _ in range(N_LSTM_LAYERS)])
	else:
		return tuple([tf.contrib.rnn.LSTMStateTuple(np.zeros([1, HIDDEN_SIZE]),np.zeros([1, HIDDEN_SIZE]))  for _ in range(N_LSTM_LAYERS)])

# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):

	layer_normalization_params = [
		"/layer_norm_basic_lstm_cell/kernel",
		"/layer_norm_basic_lstm_cell/input/gamma",
		"/layer_norm_basic_lstm_cell/input/beta",
		"/layer_norm_basic_lstm_cell/transform/gamma",
		"/layer_norm_basic_lstm_cell/transform/beta",
		"/layer_norm_basic_lstm_cell/forget/gamma",
		"/layer_norm_basic_lstm_cell/forget/beta",
		"/layer_norm_basic_lstm_cell/output/gamma",
		"/layer_norm_basic_lstm_cell/output/beta",
		"/layer_norm_basic_lstm_cell/state/gamma",
		"/layer_norm_basic_lstm_cell/state/beta"
	]
	params = ["/basic_lstm_cell/kernel",
		"/basic_lstm_cell/bias"]

	@staticmethod
	def get_weight_names(layer_normalization):
		return GameACLSTMNetwork.layer_normalization_params if layer_normalization else GameACLSTMNetwork.params

	@staticmethod
	def create_cell(n_hidden, layer_normalization):
		if layer_normalization:
			return tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden, dropout_keep_prob=1.0)
		else:
			return tf.contrib.rnn.BasicLSTMCell(n_hidden)

	def __init__(self,
							 thread_index, # -1 for global
							 device="/cpu:0" ):
		GameACNetwork.__init__(self, thread_index, device)

		scope_name = "net_" + str(self._thread_index)
		with tf.device(self._device), tf.variable_scope(scope_name) as scope:

			self.W_state_to_hidden_fc, self.b_state_to_hidden_fc = self._fc_variable([STATE_SIZE, HIDDEN_SIZE])

			# lstm
			# self.lstm = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
			self.lstm = tf.contrib.rnn.MultiRNNCell([GameACLSTMNetwork.create_cell(HIDDEN_SIZE, LAYER_NORMALIZATION) for i in range(N_LSTM_LAYERS)], state_is_tuple=True)

			# weight for policy output layer
			self.W_hidden_to_action_mean_fc, self.b_hidden_to_action_mean_fc = self._fc_variable([HIDDEN_SIZE, 1])
			# self.W_hidden_to_action_std_fc, self.b_hidden_to_action_std_fc = self._fc_variable([HIDDEN_SIZE, 1])

			# weight for value output layer
			self.W_hidden_to_value_packets_fc, self.b_hidden_to_value_packets_fc = self._fc_variable([HIDDEN_SIZE, 1])
			self.W_hidden_to_value_delay_fc, self.b_hidden_to_value_delay_fc = self._fc_variable([HIDDEN_SIZE, 1])
			self.W_hidden_to_value_duration_fc, self.b_hidden_to_value_duration_fc = self._fc_variable([HIDDEN_SIZE, 1])

			# state (input)
			self.s = tf.placeholder(PRECISION, [None, STATE_SIZE])

			# place holder for LSTM unrolling time step size.
			self.step_size = tf.placeholder(PRECISION, [1])

			self.initial_lstm_state = lstm_state_tuple()

			# Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
			# When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
			# Unrolling step size is applied via self.step_size placeholder.
			# When forward propagating, step_size is 1.
			# (time_major = False, so output shape is [batch_size, max_time, cell.output_size])

			h_fc = tf.matmul(self.s, self.W_state_to_hidden_fc) + self.b_state_to_hidden_fc
			h_fc_reshaped = tf.reshape(h_fc, [1,-1,HIDDEN_SIZE])

			lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
															self.lstm,
															h_fc_reshaped,
															initial_state = self.initial_lstm_state,
															sequence_length = self.step_size,
															time_major = False,
															scope = scope,
															dtype = PRECISION)

			lstm_outputs = tf.reshape(lstm_outputs, [-1,HIDDEN_SIZE])

			raw_pi_mean = tf.matmul(lstm_outputs, self.W_hidden_to_action_mean_fc) + self.b_hidden_to_action_mean_fc
			# raw_pi_std = tf.matmul(lstm_outputs, self.W_hidden_to_action_std_fc) + self.b_hidden_to_action_std_fc
			# policy (output)
			self.pi = (
				tf.nn.softplus(raw_pi_mean) + 1.0,
				# tf.nn.softplus(raw_pi_std) + MINIMUM_STD
				# tf.nn.sigmoid(raw_pi_std)
				# tf.constant(0.5, shape=(1,1), dtype=PRECISION)
				tf.clip_by_value(raw_pi_mean*0.01, MINIMUM_STD, float("inf"))
			)

			# value (output)
			v_packets_ = tf.nn.softplus(tf.matmul(lstm_outputs, self.W_hidden_to_value_packets_fc) + self.b_hidden_to_value_packets_fc)
			self.v_packets = tf.reshape( v_packets_, [-1] )

			v_accumulated_delay_ = tf.nn.softplus(tf.matmul(lstm_outputs, self.W_hidden_to_value_delay_fc) + self.b_hidden_to_value_delay_fc)
			self.v_accumulated_delay = tf.reshape( v_accumulated_delay_, [-1] )

			v_duration_ = tf.nn.softplus(tf.matmul(lstm_outputs, self.W_hidden_to_value_duration_fc) + self.b_hidden_to_value_duration_fc)
			self.v_duration = tf.reshape( v_duration_, [-1] )

			scope.reuse_variables()

			all_weight_names = ["multi_rnn_cell/cell_"+str(index)+item for index, item in itertools.product(range(N_LSTM_LAYERS), GameACLSTMNetwork.get_weight_names(LAYER_NORMALIZATION))]

			self.LSTM_variables = [tf.get_variable(weight_name, dtype=PRECISION) for weight_name in all_weight_names]

			self.reset_state()

	def reset_state(self):
		self.lstm_state_out = lstm_state_tuple(use_np=True)

	# def run_policy_and_value(self, sess, s_t):
	# 	# This run_policy_and_value() is used when forward propagating.
	# 	# so the step size is 1.
	# 	pi_out, v_packets_out, v_accumulated_delay_out, v_duration_out, self.lstm_state_out = sess.run(
	# 		[self.pi, self.v_packets, self.v_accumulated_delay, self.v_duration, self.lstm_state],
	# 		feed_dict = {self.s : [s_t],
	# 		self.initial_lstm_state : self.lstm_state_out,
	# 		self.step_size : [1]}
	# 	)
	# 	# pi_out: (1,3), v_out: (1)
	# 	return ((pi_out[0][0], pi_out[1][0]), (v_packets_out[0], v_accumulated_delay_out[0], v_duration_out[0]))

	def run_policy_action_and_value(self, sess, s_t):
		# This run_policy_and_value() is used when forward propagating.
		# so the step size is 1.
		pi_out, action_out, v_packets_out, v_accumulated_delay_out, v_duration_out, self.lstm_state_out = sess.run(
			[self.pi, self.chosen_action, self.v_packets, self.v_accumulated_delay, self.v_duration, self.lstm_state],
			feed_dict = {self.s : [s_t],
			self.initial_lstm_state : self.lstm_state_out,
			self.step_size : [1]}
		)
		# pi_out: (1,3), v_out: (1)
		return ((pi_out[0][0], pi_out[1][0]), action_out[0],(v_packets_out[0], v_accumulated_delay_out[0], v_duration_out[0]))

	def run_value(self, sess, s_t):
		# This run_value() is used for calculating V for bootstrapping at the
		# end of LOCAL_T_MAX time step sequence.
		# When next sequence starts, V will be calculated again with the same state using updated network weights,
		# so we don't update LSTM state here.
		prev_lstm_state_out = self.lstm_state_out
		v_packets_out, v_accumulated_delay_out, v_duration_out, _ = sess.run( [self.v_packets, self.v_accumulated_delay, self.v_duration, self.lstm_state],
								feed_dict = {self.s : [s_t],
								self.initial_lstm_state : self.lstm_state_out,
								# self.initial_lstm_state0 : self.lstm_state_out[0],
								# self.initial_lstm_state1 : self.lstm_state_out[1],
								self.step_size : [1]} )

		# roll back lstm state
		self.lstm_state_out = prev_lstm_state_out
		return (v_packets_out[0], v_accumulated_delay_out[0], v_duration_out[0])

	def run_loss(self, sess, feed_dict):
		# We don't have to roll back the LSTM state here as it is restored in the "process" function of a3c_training_thread.py anyway and because run_loss is only called there.
 		return sess.run( [self.entropy, self.skewness, self.actor_loss, self.value_loss, self.total_loss, self.distribution_mean, self.distribution_std, self.inner_distribution_mean, self.inner_distribution_std], feed_dict = feed_dict )

	def get_vars(self):
		return [self.W_state_to_hidden_fc, self.b_state_to_hidden_fc,
						self.W_hidden_to_action_mean_fc, self.b_hidden_to_action_mean_fc,
						# self.W_hidden_to_action_std_fc, self.b_hidden_to_action_std_fc,
						self.W_hidden_to_value_packets_fc, self.b_hidden_to_value_packets_fc,
						self.W_hidden_to_value_delay_fc, self.b_hidden_to_value_delay_fc,
						self.W_hidden_to_value_duration_fc, self.b_hidden_to_value_duration_fc] + self.LSTM_variables
