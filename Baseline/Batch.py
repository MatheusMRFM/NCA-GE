import numpy as np
import random
from Constants import *

class Batch():
    def __init__(self):
        self.degree = []
        self.eigen = []
        self.centrality = []
        self.id = []
        self.size = 0

    def add_data(self, d, e, c, id):
        if np.isnan(d):
            print("DEGREE IS NAN!!!")
        if np.isnan(e):
            print("EIGEN IS NAN!!!")
        if np.isnan(c):
            print("CENTRALITY IS NAN!!!")
        self.degree.append([d])
        self.eigen.append([e])
        self.centrality.append([c])
        self.id.append(id)
        self.size += 1

    def reset(self):
        self.degree = []
        self.eigen = []
        self.centrality = []
        self.id = []
        self.size = 0

    def shuffle(self):
        index = np.arange(self.size)
        random.shuffle(index)
        d_temp = []
        e_temp = []
        c_temp = []
        i_temp = []
        for i in range(0, self.size):
            d_temp.append(self.degree[index[i]])
            e_temp.append(self.eigen[index[i]])
            c_temp.append(self.centrality[index[i]])
            i_temp.append(self.id[index[i]])
        self.degree = d_temp
        self.eigen = e_temp
        self.centrality = c_temp
        self.id = i_temp
