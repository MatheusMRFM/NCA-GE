from time import sleep
from time import time
import numpy as np
import time, random, sys, threading
import multiprocessing
from Worker import *
import Constants

tf.reset_default_graph()

if len(sys.argv) > 1:
	Constants.LEARNING_RATE = float(sys.argv[1])
	Constants.INIT_WEIGHT = float(sys.argv[2])
	Constants.MODE = int(sys.argv[3])
	Constants.TEST_NUM = int(sys.argv[4])
	run = int(sys.argv[5])
	if Constants.MODE == Constants.TEST or Constants.MODE == Constants.REAL_NET:
		Constants.LOAD = True
		if Constants.MODE == Constants.REAL_NET:
			Constants.NUM_WORKER = 1
	Constants.SUMMARY_NAME        = "{:.0e}".format(Constants.LEARNING_RATE) + "_WEIGHT_" + str(Constants.INIT_WEIGHT) + "_" + str(run)
	print(Constants.LEARNING_RATE, ", ", Constants.INIT_WEIGHT, ", ", Constants.MODE)

print('\n\n', Constants.SUMMARY_NAME, '\n\n')

epochs = tf.Variable(0,dtype=tf.int32,name='epochs',trainable=False)
epochs_test = tf.Variable(0,dtype=tf.int32,name='epochs_test',trainable=False)
total_graphs = tf.Variable(0,dtype=tf.int32,name='total_graphs',trainable=False)
train_nodes = tf.Variable(0,dtype=tf.int32,name='train_nodes',trainable=False)
test_nodes = tf.Variable(0,dtype=tf.int32,name='test_nodes',trainable=False)
learning_rate = tf.train.polynomial_decay(	Constants.LEARNING_RATE,
											train_nodes,
											Constants.MAX_STEPS//2,
											Constants.LEARNING_RATE*0.1)

"""
Initializes tensorflow variables
"""
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
	with tf.device("/cpu:0"):
		summary_writer = tf.summary.FileWriter("./Summary/"+Constants.SUMMARY_NAME)
		summary = Summary(summary_writer, Constants.MODE)
		master_worker = Worker('global', session, learning_rate, epochs, epochs_test, total_graphs, train_nodes, test_nodes, summary)
		workers = []
		for i in range(Constants.NUM_WORKER):
			print (i)
			workers.append(Worker(i, session, learning_rate, epochs, epochs_test, total_graphs, train_nodes, test_nodes, summary))

	saver = tf.train.Saver(max_to_keep=1)
	if Constants.LOAD:
		print ("Loading....")
		c = tf.train.get_checkpoint_state(Constants.MODEL_PATH)
		saver.restore(session,c.model_checkpoint_path)
		print ("Graph loaded!")
	else:
		session.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()

	"""
	Initializes the worker threads
	"""
	worker_threads = []
	for i in range(Constants.NUM_WORKER):
		t = threading.Thread(target=workers[i].work, args=(coord,saver))
		t.start()
		sleep(0.5)
		worker_threads.append(t)

	coord.join(worker_threads)
