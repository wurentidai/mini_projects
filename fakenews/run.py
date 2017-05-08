#!/usr/bin/env python

# import the required packages here
import gensim
import tensorflow as tf
from tensorflow.contrib import learn
from vocab import *

num_epochs = 50
evaluate_step = 100
num_filters = 128
batch_size = 64

class CNNClassifier(object):
	"""docstring for CNNClassifier"""
	def __init__(self, sequence_length, num_classes, filter_sizes, num_filers, embedding_matrix, l2_reg_lambda=0.0):
		super(CNNClassifier, self).__init__()		
		self.embedding_size = embedding_matrix.shape[1]
		
		self.input = tf.placeholder(tf.int32, [None, sequence_length], name='input_X')
		self.label = tf.placeholder(tf.float32, [None, num_classes], name='input_Y')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		l2_loss = tf.constant(0.0)

		with tf.device('/cpu:0'),tf.variable_scope('embedding'):
			self.W = tf.Variable(embedding_matrix,name='W',dtype=tf.float32) 
			self.embedding_layer = tf.nn.embedding_lookup(self.W,self.input)
			self.embedding_layer_ = tf.expand_dims(self.embedding_layer,-1)

		pooling = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.variable_scope("conv-maxpool-%s" % filter_size):
				filter_shape = [filter_size, self.embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv2d(
					self.embedding_layer_,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooling.append(pooled)

		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooling, 3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		with tf.variable_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		with tf.variable_scope('output'):
			W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		with tf.variable_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.label)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		with tf.variable_scope('optimize'):
			self.global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-3)
			grads_and_vars = optimizer.compute_gradients(self.loss)
			self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

		with tf.variable_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	def update_step(self,data_x,data_y,dropout,sess=None):
		sess = sess or tf.get_default_session()
		_, step, loss, accuracy = sess.run([self.train_op,self.global_step, self.loss, self.accuracy], {self.input:data_x, self.label:data_y,self.dropout_keep_prob:dropout})
		return [step, loss, accuracy]

	def predict(self,data_x, data_y,sess=None):
		sess = sess or tf.get_default_session()
		loss, accuracy, predictions = sess.run([self.loss, self.accuracy, self.predictions],{self.input:data_x, self.label:data_y, self.dropout_keep_prob:1.0})
		return [loss, accuracy, predictions]

def run(train_file, valid_file, test_file, output_file):
	print 'Preparing train/dev data'
	x_train, y_train = build_data(train_file, 79)
	x_dev, y_dev = build_data(valid_file, 79)

	print 'Loading word2vec embeddings'
	embedding = np.load('embeddings.npy')

	with tf.Graph().as_default():
		conf = tf.ConfigProto(allow_soft_placement=True)
		sess = tf.Session(config = conf)
		with sess.as_default():
			classifier = CNNClassifier(x_train.shape[1], 6, [3,4,5], 128, embedding, l2_reg_lambda=0.0)
			sess.run(tf.global_variables_initializer())
			for epoch in range(num_epochs):
				batches = batch_generator(x_train,y_train, batch_size)
				for batch in batches:
					current_step, current_loss, batch_accuracy = classifier.update_step(batch[0], batch[1], 0.5, sess)
					# if current_step % evaluate_step == 0:
					# 	_1, dev_accuracy, _2 = classifier.predict(x_dev, y_dev, sess)
					# 	print '\nACCURACY ON DEV SET: %f' % dev_accuracy
				loss, accuracy, predictions = classifier.predict(x_train, y_train)
				print '\nTRAINNING ERROR & ACCURACY AFTER %d EPOCH: %f, %f' % (epoch, loss, accuracy)
				loss, accuracy, predictions = classifier.predict(x_dev, y_dev)
				print 'VALIDATION ERROR & ACCURACY AFTER %d EPOCH: %f, %f' % (epoch, loss, accuracy)

if __name__ == '__main__':
	run('liar_dataset/train.tsv', 'liar_dataset/valid.tsv', 'liar_dataset/valid.tsv', 'liar_dataset/predictions.txt')
