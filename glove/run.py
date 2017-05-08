#!/usr/bin/env python

import numpy as np
import math
import random
import time 
import tqdm
import warnings
warnings.filterwarnings("error")

dimension = 200
window_size = 15
batch_size = 128

def run(input_file, vocab_file, output_file):
	'''The function to run your ML algorithm on given datasets, generate the output and save them into the provided file path
	
	Parameters
	----------
	input_file: string
		the path to the zipped text file
        vocab_file: string
                the path to the vocabulary text file
	output_file: string
		the output word embeddings to be saved
	'''

	## your implementation here
	# read data from input
	with open(input_file) as f:
		words = f.read().split()

	word2id = {}
	id2word = {}
	with open(vocab_file) as f:
		vocab = f.read().split('\n')
		vocab = vocab[:-1]

	vocab_size = len(vocab)

	for idx,item in enumerate(vocab):
		word2id[item] = idx
		id2word[idx] = item

	X = np.zeros((vocab_size, vocab_size))
	for idx in xrange(len(words)):
		if words[idx] in word2id:
			for neighbor in xrange(idx - window_size, idx + window_size+1):
				if 0 <= neighbor <= len(words)-1 and neighbor != idx:
					if words[neighbor] in word2id:
						X[word2id[words[idx]], word2id[words[neighbor]]] += 1.0/np.fabs(neighbor - idx)

	W = ((np.random.rand(vocab_size, dimension) - 0.5) / float(dimension + 1))
	W_ = ((np.random.rand(vocab_size, dimension) - 0.5) / float(dimension + 1))
	b = ((np.random.rand(vocab_size) - 0.5) / float(dimension + 1))
	b_ = ((np.random.rand(vocab_size) - 0.5) / float(dimension + 1))	
	f_vec = np.vectorize(f_)
	F_X = f_vec(X)
	non_zeros = np.nonzero(X)
	training_data = zip(non_zeros[0], non_zeros[1])
	num_batch = len(training_data)/batch_size + 1

	# your training algorithm
	learning_rate = 0.03
	for epoch in xrange(100):

		print 'Epoch %d' % epoch
		random.shuffle(training_data)

		for batch_index in tqdm.tqdm(xrange(num_batch)):
			# print 'Batch No.%d of Epoch:%d' %(batch_index, epoch)
			batch = get_batch(batch_index, batch_size, training_data)
			# print 'Loss before update: ', batch_loss(batch, F_X, X, W, W_, b, b_)
			J = {}
			updates_W = set()
			updates_W_ = set()
			for sample in batch:
				i = sample[0]
				j = sample[1]
				updates_W.add(i)
				updates_W_.add(j)
				try:
					J[(i,j)] = 2*F_X[i,j]*(np.sum(W[i,:]*W_[j,:]) + b[i] + b_[j] - np.log(X[i,j]))
				except RuntimeWarning:
					pass
					# print 'Batch%d Plance0 ' % batch_index, X[i,j], np.sum(W[i,:]*W_[j,:]), F_X[i,j]
				# else:
				# 	J[(i,j)] += 2*F_X[i,j]*(np.sum(W[i,:]*W_[j,:]) + b[i] + b_[j] - np.log(X[i,j]))
			new_W = W.copy()
			new_b = b.copy()
			new_W_ = W_.copy()
			new_b_ = b_.copy()
			for index in updates_W:
				delta = np.zeros(dimension)
				delta_b = 0
				count = 0
				for item in J.keys():
					if item[0] == index:
						count += 1
						try:
							delta += J[item]*W_[item[1],:]
							delta_b += J[item]
						except RuntimeWarning:
							print 'Place1 J value X value', J[item], X[item[0],item[1]]
						
				new_W[index,:] = W[index,:] - learning_rate*delta
				new_b[index] = b[index] - learning_rate*delta_b

			for index_ in updates_W_:
				delta = np.zeros(dimension)
				delta_b_ = 0
				count = 0
				for item in J.keys():
					if item[1] == index_:
						count += 1
						try:
							delta += J[item]*W[item[0],:]
						except RuntimeWarning:
							print 'Place2 J value, X value', J[item], X[item[0],item[1]]					
						delta_b_ += J[item]
				new_W_[index_,:] = W_[index_,:] - learning_rate*delta
				new_b_[index_] = b_[index_] = learning_rate*delta_b_
			W = new_W.copy()
			b = new_b.copy()
			W_ = new_W_.copy()
			b_ = new_b_.copy()
		# W = W / (np.linalg.norm(W, axis = -1).reshape(-1,1))
		# W_ = W_ / (np.linalg.norm(W_, axis = -1).reshape(-1,1))
		if (epoch + 1) == 50:
			learning_rate = 2*learning_rate
		#	saveEmbed(W+W_, '50' + output_file, id2word)
		#	print 'Loss after epoch%d: %f' % (epoch, batch_loss(training_data, F_X, X, W, W_, b, b_))
		# learning_rate = learning_rate/
	# save your word vectors into the file output_file
	saveEmbed(W+W_, '200_15_128_0.03_shuffle'+output_file, id2word)


# define other functions here
def batch_loss(batch, F_X, X, W , W_, b, b_):
	loss = 0
	for sample in batch:
		i = sample[0]
		j = sample[1]
		loss += F_X[i,j]*(np.sum(W[i,:]*W_[j,:]) + b[i] + b_[j] - np.log(X[i,j]))**2
	return loss


def saveEmbed(W, output_file, id2word):
	f = open(output_file, 'w')
	for idx in xrange(len(id2word.keys())):
		line = id2word[idx] + ' ' + ' '.join(map(lambda x:str(x), W[idx,:])) + ' \n'
		f.write(line)
	f.close()

def f_(x):
	if x < 100:
		return (x/100.0)**(3/4.0)
	else:
		return 1

def get_batch(batch, batch_size, training_data):
	total_size = len(training_data)
	if batch_size*(batch+1) <= total_size:
		return training_data[(batch*batch_size):(batch_size*(batch+1))]
	else:
		return training_data[batch*batch_size:]

if __name__ == "__main__":
	run('text8', 'vocab.txt', 'vectors.txt')