#!/usr/bin/env python

from collections import Counter
import csv
import re
import gensim
import numpy as np
from tensorflow.contrib import learn

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def build_vocab():
	train_path = 'liar_dataset/train.tsv'
	train_data = open(train_path).readlines()
	words = []
	for sample in train_data:
		label = sample.split('\t')[0]
		sentence_ = sample.split('\t')[1][:-1]
		sentence = clean_str(sentence_)
		tokens = sentence.split()
		words += tokens

	words_stats = Counter(words).items()
	words_stats = sorted(words_stats, key = lambda x:x[1], reverse=True)

	g = open('vocab_stats.txt','w')
	for item in words_stats:
		g.write(item[0] + '\t' + str(item[1]) + '\n')
	g.close()


def build_embedding():
	model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True)
	print model.wv[clean_str('foreign-policy')]
	# train_data = open('liar_dataset/valid.tsv').readlines()
	# x_text = []
	# for sample in train_data:
	# 	sentence = clean_str(sample.split('\t')[1][:-1])
	# 	x_text.append(sentence)
	# max_length = max([len(x.split()) for x in x_text])
	# print max_length
	# vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
	# x = np.array(list(vocab_processor.fit_transform(x_text)))
	# word2id = vocab_processor.vocabulary_._mapping
	# sorted_vocab = sorted(word2id.items(), key = lambda x:x[1])
	# embeddings = []
	# for (word,id_) in sorted_vocab:
	# 	try:
	# 		embeddings.append(model.wv[word])
	# 	except Exception as e:
	# 		embeddings.append(np.random.rand(300) - 0.5)
	# np.save('embeddings.npy', embeddings)

def meta2vec(model, words):
	vectors = []
	for word in words:
		try:
			vectors.append(model.wv[word])
		except Exception as e:
			vectors.append(np.random.rand(300) - 0.5)
	if len(vectors) == 0:
		vectors.append(np.zeros(300))
	return np.sum(vectors,0).reshape((1,-1))

def build_data(data_path, max_length, model, test=False):
	if test == True:
		data = open(data_path).readlines()
		x_text = []
		x_meta = []
		x_lens = []
		for sample in data:
			sentence = clean_str(sample.split('\t')[0][:-1])
			subjects = meta2vec(model,clean_str(sample.split('\t')[1]).split())
			speaker = meta2vec(model,clean_str(sample.split('\t')[2]).split())
			job = meta2vec(model,clean_str(sample.split('\t')[3]).split())
			state = meta2vec(model,clean_str(sample.split('\t')[4]).split())
			party = meta2vec(model,clean_str(sample.split('\t')[5]).split())
			context = meta2vec(model,clean_str(sample.split('\t')[6]).split())
			meta = np.concatenate((subjects, speaker,job,state,party,context), axis=0)
			x_meta.append(meta)
			x_text.append(sentence)
			x_lens.append(len(sentence.split()))
		vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
		x = np.array(list(vocab_processor.fit_transform(x_text)))
		x_meta = np.array(x_meta)
		x_lens = np.array(x_lens)
		return [x, x_meta, x_lens]

	label_dict = {'pants-fire':0,'false':1,'barely-true':2,'half-true':3,'mostly-true':4,'true':5}
	data = open(data_path).readlines()
	x_text = []
	labels = []
	x_meta = []
	x_lens = []
	for sample in data:
		labels.append(label_dict[sample.split('\t')[0]])
		sentence = clean_str(sample.split('\t')[1][:-1])
		subjects = meta2vec(model,clean_str(sample.split('\t')[2]).split())
		speaker = meta2vec(model,clean_str(sample.split('\t')[3]).split())
		job = meta2vec(model,clean_str(sample.split('\t')[4]).split())
		state = meta2vec(model,clean_str(sample.split('\t')[5]).split())
		party = meta2vec(model,clean_str(sample.split('\t')[6]).split())
		context = meta2vec(model,clean_str(sample.split('\t')[7]).split())
		meta = np.concatenate((subjects, speaker,job,state,party,context), axis=0)
		x_meta.append(meta)
		x_text.append(sentence)
		x_lens.append(len(sentence.split()))

	labels_onehot = np.zeros((len(labels),6))
	labels_onehot[np.arange(len(labels)),np.array(labels)] = 1
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
	x = np.array(list(vocab_processor.fit_transform(x_text)))
	x_meta = np.array(x_meta)
	x_lens = np.array(x_lens)
	return [x, x_meta, x_lens, labels_onehot]

def batch_generator(data_x, data_x_meta, data_lens, data_y, batch_size, shuffle=True):
	data_size = data_x.shape[0]
	num_batches = int((data_size - 1)/batch_size) + 1
	if shuffle:
		shuffle_indices = np.random.permutation(np.arange(data_size))
		data_x_shuffle = data_x[shuffle_indices]
		data_x_meta_shuffle = data_x_meta[shuffle_indices]
		data_y_shuffle = data_y[shuffle_indices]
		data_lens_shuffle = data_lens[shuffle_indices]
	else:
		data_x_shuffle = data_x
		data_y_shuffle = data_y
		data_x_meta_shuffle = data_x_meta
		data_lens_shuffle = data_lens
	for batch in range(num_batches):
		start_index = batch * batch_size
		end_index = min((batch + 1) * batch_size, data_size)
		yield [data_x_shuffle[start_index:end_index],data_x_meta_shuffle[start_index:end_index], data_lens_shuffle[start_index:end_index],data_y_shuffle[start_index:end_index]]

if __name__ == '__main__':
	#build_embedding()
	build_data('liar_dataset/valid.tsv', 79)
