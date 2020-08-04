import networkx as nx
import numpy as np
import os
from time import sleep
from time import time
import numpy as np
import time, random, threading
import multiprocessing
from Graph import *

num_workers = 6

SMALLWORLD  = 0
SCALEFREE   = 1
RANDOM      = 2
SCALE_SMALL	= 3
MIX         = 4

TRAIN   = 0
TEST    = 1

"""
Variables that control how the graphs are generated
"""
NUM_TRAIN_GRAPH		= 0
NUM_TEST_GRAPH		= 100

GRAPH_TYPE 			= 3
MAX_NODES			= 1000
MIN_NODES			= 100
MAX_P				= 0.8
MIN_P 				= 0.1
MAX_K				= 8
MIN_K 				= 2
MAX_M				= 10
MIN_M 				= 3

"""
Variables for controlling race conditions
"""
lock = threading.Lock()
graph_train_number = -1
graph_test_number = -1

class Graph_Generator():
	def __init__(self, id, num_worker):
		self.num_worker = num_worker
		self.tid = id
		self.folder_name = "./Graphs/"

	#---------------------------------------------------------------------------
	def generate_set(self, mode):
		global graph_train_number
		global graph_test_number

		if mode == TRAIN:
			NUM_IT = int(NUM_TRAIN_GRAPH/self.num_worker)
			sub_folder = "Train/"
		else:
			NUM_IT = int(NUM_TEST_GRAPH/self.num_worker)
			sub_folder = "Test_4/"

		graph = Graph()

		for i in range(0, NUM_IT):
			lock.acquire()
			if mode == TRAIN:
				graph_train_number += 1
				g_number = graph_train_number
			else:
				graph_test_number += 1
				g_number = graph_test_number
			lock.release()

			n, p, k, m, type = self.fetch_variables()

			if type == RANDOM:
				graph.generate_random_graph(n, p)
			elif type == SMALLWORLD:
				graph.generate_smallworld_graph(n, p, k)
			elif type == SCALEFREE:
				graph.generate_scalefree_graph(n, m)
			else:
				print("Unexpected GRAPH_TYPE value.")

			graph.save_file(g_number, self.folder_name, sub_folder)

	#---------------------------------------------------------------------------
	def fetch_variables(self):
		n = np.random.randint(MIN_NODES, MAX_NODES+1)
		p = float(np.random.randint(int(1000*MIN_P), int(1000*(MAX_P)))) / 1000.0
		k = np.random.randint(MIN_K, MAX_K+1)
		m = np.random.randint(MIN_M, MAX_M+1)

		type = GRAPH_TYPE
		if GRAPH_TYPE == MIX:
			type = np.random.randint(0, RANDOM+1)
		elif GRAPH_TYPE == SCALE_SMALL:
			type = np.random.randint(0, SCALEFREE+1)

		return n, p, k, m, type

	#---------------------------------------------------------------------------
	def run_graph_generator(self):
		#self.generate_set(TRAIN)
		self.generate_set(TEST)






workers = []
for i in range(num_workers):
	print (i)
	workers.append(Graph_Generator(i, num_workers))

"""
Initializes the worker threads
"""
worker_threads = []
for i in range(num_workers):
	t = threading.Thread(target=workers[i].run_graph_generator, args=())
	t.start()
	sleep(0.5)
	worker_threads.append(t)
