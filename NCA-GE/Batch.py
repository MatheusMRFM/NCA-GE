import numpy as np
import random
import scipy as sp
from Constants import *

class Batch():
    def __init__(self, a):
        self.degree = None
        self.features = None
        self.adj = a
        self.centrality = []
        self.id = []
        self.size = 0
        self.num_nodes = None

    def add_data(self, d, f, c, id):
        if self.num_nodes is None:
            self.num_nodes = len(f)
        if self.degree is None:
            self.degree = d
        if self.features is None:
            self.features = f
        self.centrality.append([c])
        self.id.append(id)
        self.size += 1

    def reset(self):
        self.features = []
        self.adj = None
        self.centrality = []
        self.id = []
        self.size = 0
        self.num_nodes = None

    def shuffle(self):
        index = np.arange(self.size)
        random.shuffle(index)
        c_temp = []
        i_temp = []
        for i in range(0, self.size):
            c_temp.append(self.centrality[index[i]])
            i_temp.append(self.id[index[i]])
        self.centrality = c_temp
        self.id = i_temp
