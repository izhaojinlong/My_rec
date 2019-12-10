#coding=utf-8
import os
import sys

train_rate = 0.8
dim = 8
walk_len = 5
win_size = 3
num_walk = 10
'''Movielens dataset meta-path'''
metapaths = ['umu', 'umgmu', 'uou', 'uau', 'mum']

for metapath in metapaths:
	metapath = metapath + '_' + str(train_rate)
	input_file = '../data/Movielens/metapath/' + metapath + '.txt'
	output_file = '../data/Movielens/embeddings/' + metapath + '.embedding'

	cmd = 'deepwalk --format edgelist --input ' + input_file + ' --output ' + output_file + \
	      ' --walk-length ' + str(walk_len) + ' --window-size ' + str(win_size) + ' --number-walks '\
	       + str(num_walk) + ' --representation-size ' + str(dim)

	print (u''+cmd)
	os.system(cmd)
