from time import sleep
from time import time
import numpy as np
import time, random, threading, sys
import multiprocessing
from Worker import *
import Constants
import os

tf.reset_default_graph()

if len(sys.argv) > 1:
	Constants.MODE = int(sys.argv[1])
	Constants.EMBED_METHOD = int(sys.argv[2])
	Constants.LAYERS = int(sys.argv[3])
	Constants.LEARNING_RATE = float(sys.argv[4])
	Constants.MIN_LEARNING_RT = float(sys.argv[5])
	Constants.NUM_FEATURES = int(sys.argv[6])
	Constants.INPUT = int(sys.argv[7])
	Constants.TARGET = int(sys.argv[8])
	Constants.RANGE = int(sys.argv[9])
	Constants.BATCH_SIZE = int(sys.argv[10])
	Constants.EMBED_SIZE = int(sys.argv[11])
	Constants.LAMBDA_REGUL = float(sys.argv[12])
	Constants.TH_CENT = float(sys.argv[13])
	Constants.TEST_NUM = int(sys.argv[14])

	IN_RANK_NAME = 'RANK'
	if Constants.INPUT == Constants.VALUE:
		IN_RANK_NAME = 'VALUE'

	OUT_RANK_NAME = 'RANK'
	if Constants.TARGET == Constants.VALUE:
		OUT_RANK_NAME = 'VALUE'

	METHOD_NAME = 'GCN'
	if Constants.EMBED_METHOD == Constants.S2VEC:
		METHOD_NAME = 'S2VEC'

	RANGE_NAME = '01'
	if Constants.RANGE == Constants.RANGE_11:
		RANGE_NAME = '11'

	if Constants.MODE == Constants.TEST or Constants.MODE == Constants.REAL_NET:
		Constants.LOAD = True
		if Constants.MODE == Constants.REAL_NET:
			Constants.NUM_WORKER = 1

	Constants.SUMMARY_NAME = METHOD_NAME + NORM_NAME + \
							"_LAYER=" + str(Constants.LAYERS) + \
							"_LR=" + "{:.0e}".format(Constants.LEARNING_RATE) + "-" + \
							"{:.0e}".format(Constants.MIN_LEARNING_RT) + \
							"_F=" + str(Constants.NUM_FEATURES) + \
							"_IN=" + IN_RANK_NAME + \
							"_OUT=" + OUT_RANK_NAME + \
							"_RANGE=" + RANGE_NAME + \
							"_BATCH=" + str(Constants.BATCH_SIZE) + \
							"_EMBED=" + str(Constants.EMBED_SIZE) + \
							"_LAMBDA=" + str(Constants.LAMBDA_REGUL) + \
							"_TH=" + "{:.2f}".format(Constants.TH_CENT) + \
                       		"_" + str(Constants.TEST_NUM)

	Constants.MODEL_PATH      = "./Model/" + Constants.SUMMARY_NAME + '/'

print('\n\n', Constants.SUMMARY_NAME, '\n\n')

epochs = tf.Variable(0,dtype=tf.int32,name='epochs',trainable=False)
epochs_test = tf.Variable(0,dtype=tf.int32,name='epochs_test',trainable=False)
total_graphs = tf.Variable(0,dtype=tf.int32,name='total_graphs',trainable=False)
train_nodes = tf.Variable(0,dtype=tf.int32,name='train_nodes',trainable=False)
test_nodes = tf.Variable(0,dtype=tf.int32,name='test_nodes',trainable=False)
learning_rate = tf.train.polynomial_decay(	Constants.LEARNING_RATE,
											train_nodes,
											Constants.MAX_STEPS//2,
											Constants.LEARNING_RATE*0.01)

"""
Initializes tensorflow variables
"""
os.environ["CUDA_VISIBLE_DEVICES"]='0'
#config = tf.ConfigProto()
config = tf.ConfigProto(device_count={"CPU":4})
config.intra_op_parallelism_threads=4
config.inter_op_parallelism_threads=4
config.allow_soft_placement=True 
config.log_device_placement=False
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
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
