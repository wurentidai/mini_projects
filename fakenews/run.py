#!/usr/bin/env python

# import the required packages here
import gensim
import tensorflow as tf
from tensorflow.contrib import learn, rnn
from vocab import *

num_epochs = 20
evaluate_step = 100
num_filters = 128
batch_size = 64

class Classifier(object):
	"""docstring for Classifier"""
	def __init__(self, num_layers, max_seq_len, num_classes, filter_sizes, num_filers, embedding_matrix, n_hidden, l2_reg_lambda=0.0):
		super(Classifier, self).__init__()

		### input data
		self.input_seq = tf.placeholder(tf.int32, [None,max_seq_len], name='input_seq')
		self.seq_len = tf.placeholder(tf.int32, [None], name='seq_length')

		self.input_meta = tf.placeholder(tf.float32, [None, 6, 300], name='input_meta') # Encode the metadata as a matrix
		self.label = tf.placeholder(tf.int32, [None, num_classes], name='input_y')

		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		with tf.device('/cpu:0'),tf.variable_scope('embedding'):
			self.W = tf.Variable(embedding_matrix,name='W',dtype=tf.float32)
			self.lstm_input = tf.nn.embedding_lookup(self.W, self.input_seq)

		self.embedding_layer_ = tf.expand_dims(self.input_meta,-1)

		### Multi-layer LSTM for statement
		cells = []
		for _ in range(num_layers):
			cells.append(tf.contrib.rnn.BasicLSTMCell(n_hidden))

		self.cell = rnn.MultiRNNCell(cells)
		self.lstm_input = tf.unstack(self.lstm_input, max_seq_len, 1)
		lstm_outputs,_ = tf.contrib.rnn.static_rnn(self.cell, self.lstm_input, sequence_length=self.seq_len, dtype=tf.float32)
		lstm_outputs = tf.transpose(tf.stack(lstm_outputs), [1,0,2])
		batch_size = tf.shape(lstm_outputs)[0]
		index = tf.range(0,batch_size) * max_seq_len + (self.seq_len - 1)
		self.lstm_outputs = tf.gather(tf.reshape(lstm_outputs, [-1, n_hidden]), index)

		l2_loss = tf.constant(0.0)

		### CNN for meta data
		pooling = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.variable_scope("conv-maxpool-%s" % filter_size):
				filter_shape = [filter_size, embedding_matrix.shape[1], 1, num_filters]
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
					ksize=[1, 6 - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooling.append(pooled)

		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooling, 3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		### Concatenate CNN and LSTM features
		self.features = tf.concat([self.lstm_outputs, self.h_pool_flat], axis=1)

		with tf.variable_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.features, self.dropout_keep_prob)

		with tf.variable_scope('output'):
			W = tf.Variable(tf.truncated_normal([num_filters_total + n_hidden, num_classes], stddev=0.1), name="W")
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
			optimizer = tf.train.AdamOptimizer(1e-4)
			grads_and_vars = optimizer.compute_gradients(self.loss)
			self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

		with tf.variable_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	def update_step(self,data_x, data_meta, data_lens, data_y,dropout,sess=None):
		sess = sess or tf.get_default_session()
		_, step, loss, accuracy = sess.run([self.train_op,self.global_step, self.loss, self.accuracy], {self.input_seq:data_x, self.input_meta:data_meta, self.seq_len:data_lens ,self.label:data_y,self.dropout_keep_prob:dropout})
		return [step, loss, accuracy]

	def predict(self,data_x, data_meta, data_lens, data_y,sess=None):
		sess = sess or tf.get_default_session()
		loss, accuracy, predictions = sess.run([self.loss, self.accuracy, self.predictions],{self.input_seq:data_x, self.input_meta:data_meta, self.seq_len:data_lens ,self.label:data_y, self.dropout_keep_prob:1.0})
		return [loss, accuracy, predictions]

	def test(self, data_x, data_meta, data_lens, output_file, sess=None):
		sess = sess or tf.get_default_session()
		predictions = sess.run(self.predictions, {self.input_seq:data_x, self.input_meta:data_meta, self.seq_len:data_lens, self.dropout_keep_prob:1.0})
		label_dict = {0:'pants-fire',1:'false',2:'barely-true',3:'half-true',4:'mostly-true',5:'true'}
		g = open(output_file,'w')
		for item in predictions:
			g.write(label_dict[item] + '\n')
		g.close()

def run(train_file, valid_file, test_file, output_file):
	print 'Preparing train/dev data'
	model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True)
	x_train, x_train_meta, x_train_lens, y_train = build_data(train_file, 79, model)
	x_dev, x_dev_meta, x_dev_lens, y_dev = build_data(valid_file, 79, model)
	x_test, x_test_meta, x_test_lens = build_data(test_file, 79, model,test=True)

	print 'Loading word2vec embeddings'
	embedding = np.load('embeddings.npy')

	with tf.Graph().as_default():
		conf = tf.ConfigProto(allow_soft_placement=True)
		sess = tf.Session(config = conf)
		with sess.as_default():
			classifier = Classifier(1, 79, 6, [5,3,4], 64, embedding, 300, l2_reg_lambda=0.3)
			sess.run(tf.global_variables_initializer())
			for epoch in range(num_epochs):
				batches = batch_generator(x_train,x_train_meta, x_train_lens,y_train, batch_size, shuffle=True)
				for batch in batches:
					current_step, current_loss, batch_accuracy = classifier.update_step(batch[0], batch[1], batch[2], batch[3], 0.5, sess)
					# if current_step % evaluate_step == 0:
					# 	_1, dev_accuracy, _2 = classifier.predict(x_dev, y_dev, sess)
					# 	print '\nACCURACY ON DEV SET: %f' % dev_accuracy
				loss, accuracy, predictions = classifier.predict(x_train, x_train_meta, x_train_lens, y_train)
				print '\nTRAINNING ERROR & ACCURACY AFTER %d EPOCH: %f, %f' % (epoch, loss, accuracy)
				loss, accuracy, predictions = classifier.predict(x_dev, x_dev_meta, x_dev_lens, y_dev)
				print 'VALIDATION ERROR & ACCURACY AFTER %d EPOCH: %f, %f' % (epoch, loss, accuracy)
				print 'SAVEING RESULTS'
				classifier.test(x_test, x_test_meta, x_test_lens, str(epoch) + output_file)


if __name__ == '__main__':
	run('liar_dataset/train.tsv', 'liar_dataset/valid.tsv', 'liar_dataset/test.tsv', 'predictions.txt')
