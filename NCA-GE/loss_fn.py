import tensorflow as tf
import scipy as sp
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layer
import tensorflow.keras.backend as K
import time, random, threading
import numpy as np
import gc

EPSILON = 1e-20

'''
Implementação de funções de perda que otimizam o ranking ao invés
de otimizar o erro direto. Ver o seguinte artigo para mais detalhes:
- "SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS"
'''

def bpr_loss(x_pos, x_neg):
	return -tf.reduce_mean(tf.log(tf.sigmoid(x_pos - x_neg) + EPSILON))

def top1_loss(x_pos, x_neg):
	return tf.reduce_mean(tf.sigmoid(x_neg - x_pos) + tf.sigmoid(x_neg))
