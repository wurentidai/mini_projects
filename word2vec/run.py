#!/usr/bin/env python

import numpy as np
import math
import random
import time

num_noise = 5
dimension = 200
window_size = 10

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
	print 'PREPROCESSING'

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

	word_count = np.zeros(vocab_size)
	for word in words:
		if word in word2id:
			word_count[word2id[word]] += 1
	word_p = word_count**(0.75)
	word_f = word_p / np.sum(word_p)


	training_data = []
	ids = range(vocab_size)
	for idx in range(len(words)):
		if words[idx] in word2id:
			for neighbor in range(idx - window_size, idx + window_size+1):
				if 0 <= neighbor <= len(words)-1 and neighbor != idx and words[neighbor] in word2id:
					center = word2id[words[idx]]
					target = word2id[words[neighbor]]
					negative = []
					noises = np.random.choice(ids, num_noise, p = word_f ,replace=False)
					for noise in noises:
						if noise != target:
							negative.append(noise)
					training_data.append((center, target, negative))


	# your training algorithm
	W = np.random.uniform(-0.5 / dimension,0.5 / dimension,(vocab_size, dimension))
	W_out = np.zeros((dimension, vocab_size))
	learning_rate = 0.05

	# sigmoid_vec = np.vectorize(sigmoid)

	# train_size = len(training_data)
	print 'BEGIN TRAINING'
	for epoch in xrange(50):
		# random.shuffle(training_data)
		print 'Epoch%d' % epoch
		for index, sample in enumerate(training_data):
			
			center = sample[0]
			target = sample[1]
			noise_elems = sample[2]
			# embed_c = np.matmul(one_hot(center, vocab_size), W)
			embed_c = W[center,:]
			#scores = np.squeeze(sigmoid_vec(np.matmul(embed_c, W_out)))

			# updating step
			update_elems = noise_elems + [target]
			EH = 0
			for idx in update_elems:
				score = sigmoid(np.sum(embed_c*W_out[:,idx]))
				EI = score - (idx ==target)
				EH += EI*W_out[:,idx]
				W_out[:,idx] = W_out[:,idx] - learning_rate*EI*W[center,:]
			W[center,:] = W[center,:] - learning_rate*EH

		# learning_rate = learning_rate/2.0
	# save your word vectors into the file output_file
	saveEmbed(W, '200_epoch50_window10'+output_file, id2word)


# define other functions here
def saveEmbed(W, output_file, id2word):
	f = open(output_file, 'w')
	for idx in xrange(len(id2word.keys())):
		line = id2word[idx] + ' ' + ' '.join(map(lambda x:str(x), W[idx,:])) + ' \n'
		f.write(line)
	f.close()

def sigmoid(x):
	if x > 6:
		return 1.0
	elif x < -6:
		return 0.0
	else:
		return 1.0/(1.0 + math.exp(-x))

def one_hot(x, vocab_size):
	vector = np.zeros(vocab_size)
	vector[x] = 1
	return np.expand_dims(vector, 0)

def negative_sample(sample, vocab_size, word_neighbors, num_noise):
	center = sample[0]
	context_words = word_neighbors[center]
	noise_elems = []
	for _ in xrange(num_noise):
		while True:
			noise = random.randrange(vocab_size)
			if noise not in context_words:
				noise_elems.append(noise)
				break
	return noise_elems

if __name__ == "__main__":
	start = time.time()
	run('text8', 'vocab.txt', 'vectors.txt')
	end = time.time()
	print end - start