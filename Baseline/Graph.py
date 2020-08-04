import networkx as nx
import numpy as np
import os
import Constants

class Graph():
	def __init__(self):
		self.graph = nx.Graph()

	#---------------------------------------------------------------------------
	def define_node_sample(self):
		self.sample = []
		count = 0
		ratio = 2
		current_ratio = 0
		count_reverse = self.get_num_nodes()-1
		low_idx_reverse = count_reverse
		turn_reverse = True
		while count < low_idx_reverse and count < count_reverse:
			if turn_reverse:
				node = self.cent_rank[count_reverse]
				if self.norm_cent[node] < Constants.TH_CENT:
					low_idx_reverse = count_reverse
					count_reverse = self.get_num_nodes()-1
					node = self.cent_rank[count_reverse]
				count_reverse -= 1
				current_ratio += 1
				if current_ratio > ratio:
					current_ratio = 0
					turn_reverse = not turn_reverse
			else:
				node = self.cent_rank[count]
				count += 1
				turn_reverse = not turn_reverse
			self.sample.append(node)
		self.current_id_sample = 0

	#---------------------------------------------------------------------------
	def get_num_nodes(self):
		return self.graph.number_of_nodes()

	#---------------------------------------------------------------------------
	def get_num_samples(self):
		return len(self.sample)

	#---------------------------------------------------------------------------
	def get_degree(self, i):
		return self.norm_degree[i]

	#---------------------------------------------------------------------------
	def get_centrality(self, i):
		return self.norm_cent[i]

	#---------------------------------------------------------------------------
	def get_eigenvector_centrality(self, i):
		return self.norm_eigen[i]

	#---------------------------------------------------------------------------
	def get_data_next_node(self):
		node = self.sample[self.current_id_sample]
		d = self.get_degree(node)
		e = self.get_eigenvector_centrality(node)
		c = self.get_centrality(node)
		self.current_id_sample += 1
		graph_finished = False
		if self.current_id_sample >= self.get_num_samples():
			graph_finished = True
		return d, e, c, node, graph_finished

	#---------------------------------------------------------------------------
	def predicted_values(self, batch, cent):
		for i in range(0, batch.size):
			self.predictions[batch.id[i]] = cent[i]

	#---------------------------------------------------------------------------
	def _normalize_array(self, true_value):
		rank = np.argsort(true_value, kind='mergesort', axis=None)
		max = np.amax(true_value)
		min = np.amin(true_value)
		norm = np.empty([self.get_num_nodes()])
		if max > 0.0 and max > min:
			for i in range(0, self.get_num_nodes()):
				norm[i] = 2.0*(float(true_value[i] - min) / float(max - min)) - 1.0
		else:
			print("Max, Min = (", max, ", ", min, ")")

		return norm, rank

	#---------------------------------------------------------------------------
	def _normalize_array_by_rank(self, true_value):
		rank = np.argsort(true_value, kind='mergesort', axis=None)
		norm = np.empty([self.get_num_nodes()])
		for i in range(0, self.get_num_nodes()):
			norm[rank[i]] = float(i+1) / float(self.get_num_nodes())
		max = np.amax(norm)
		min = np.amin(norm)
		if max > 0.0 and max > min:
			for i in range(0, self.get_num_nodes()):
				norm[i] = 2.0*(float(norm[i] - min) / float(max - min)) - 1.0
		else:
			print("Max value = 0")

		return norm, rank

	#---------------------------------------------------------------------------
	def normalized_degree_rank(self):
		degree = [val for (node, val) in self.graph.degree()]
		self.norm_degree, _ = self._normalize_array_by_rank(degree)

	#---------------------------------------------------------------------------
	def normalized_betweenness_rank(self):
		b = [v for v in nx.betweenness_centrality(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array_by_rank(b)

	#---------------------------------------------------------------------------
	def normalized_closeness_rank(self):
		c = [v for v in nx.closeness_centrality(self.graph, wf_improved=False).values()]
		self.norm_cent, self.cent_rank = self._normalize_array_by_rank(c)

	#---------------------------------------------------------------------------
	def normalized_eigenvector_rank(self):
		e = [v for v in nx.eigenvector_centrality_numpy(self.graph).values()]
		self.norm_eigen, self.eigen_rank = self._normalize_array_by_rank(e)

	#---------------------------------------------------------------------------
	def normalized_clustering_coef_rank(self):
		c = [v for v in nx.clustering(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array_by_rank(c)

	#---------------------------------------------------------------------------
	def normalized_flow_rank(self):
		c = [v for v in nx.current_flow_betweenness_centrality(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array_by_rank(c)

	#---------------------------------------------------------------------------
	def normalized_second_order_rank(self):
		c = [v for v in nx.second_order_centrality(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array_by_rank(c)

	#---------------------------------------------------------------------------
	def normalized_harmonic_rank(self):
		c = [v for v in nx.harmonic_centrality(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array_by_rank(c)

	#---------------------------------------------------------------------------
	def normalized_load_cent_rank(self):
		c = [v for v in nx.load_centrality(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array_by_rank(c)

	#---------------------------------------------------------------------------
	def normalized_degree(self):
		self.degree = [val for (node, val) in self.graph.degree()]
		self.norm_degree, _ = self._normalize_array(self.degree)

	#---------------------------------------------------------------------------
	def normalized_betweenness(self):
		b = [v for v in nx.betweenness_centrality(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array(b)

	#---------------------------------------------------------------------------
	def normalized_closeness(self):
		c = [v for v in nx.closeness_centrality(self.graph, wf_improved=False).values()]
		self.norm_cent, self.cent_rank = self._normalize_array(c)

	#---------------------------------------------------------------------------
	def normalized_eigenvector(self):
		e = [v for v in nx.eigenvector_centrality_numpy(self.graph).values()]
		self.norm_eigen, _ = self._normalize_array(e)

	#---------------------------------------------------------------------------
	def normalized_clustering_coef(self):
		c = [v for v in nx.clustering(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array(c)

	#---------------------------------------------------------------------------
	def normalized_flow(self):
		c = [v for v in nx.current_flow_betweenness_centrality(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array(c)

	#---------------------------------------------------------------------------
	def normalized_second_order(self):
		c = [v for v in nx.second_order_centrality(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array(c)

	#---------------------------------------------------------------------------
	def normalized_harmonic(self):
		c = [v for v in nx.harmonic_centrality(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array(c)

	#---------------------------------------------------------------------------
	def normalized_load_cent(self):
		c = [v for v in nx.load_centrality(self.graph).values()]
		self.norm_cent, self.cent_rank = self._normalize_array(c)

	#---------------------------------------------------------------------------
	def normalized_centralities_rank(self):
		self.normalized_degree_rank()
		self.normalized_eigenvector_rank()
		if Constants.CENTRALITY == Constants.BETWEEN:
			self.normalized_betweenness_rank()
		elif Constants.CENTRALITY == Constants.CLOSE:
			self.normalized_closeness_rank()
		elif Constants.CENTRALITY == Constants.EIGEN:
			self.norm_cent =  self.norm_eigen
			self.cent_rank = self.eigen_rank
		elif Constants.CENTRALITY == Constants.CLUSTER_COEF:
			self.normalized_clustering_coef_rank()
		elif Constants.CENTRALITY == Constants.FLOW:
			self.normalized_flow_rank()
		elif Constants.CENTRALITY == Constants.SECOND_ORDER:
			self.normalized_second_order_rank()
		elif Constants.CENTRALITY == Constants.HARMONIC:
			self.normalized_harmonic_rank()
		elif Constants.CENTRALITY == Constants.LOAD_CENT:
			self.normalized_load_cent_rank()

	#---------------------------------------------------------------------------
	def normalized_centralities(self):
		self.normalized_degree()
		self.normalized_eigenvector()
		if Constants.CENTRALITY == Constants.BETWEEN:
			self.normalized_betweenness()
		elif Constants.CENTRALITY == Constants.CLOSE:
			self.normalized_closeness()
		elif Constants.CENTRALITY == Constants.EIGEN: 
			self.norm_cent =  self.norm_eigen
		elif Constants.CENTRALITY == Constants.CLUSTER_COEF:
			self.normalized_clustering_coef()
		elif Constants.CENTRALITY == Constants.FLOW:
			self.normalized_flow()
		elif Constants.CENTRALITY == Constants.SECOND_ORDER:
			self.normalized_second_order()
		elif Constants.CENTRALITY == Constants.HARMONIC:
			self.normalized_harmonic()
		elif Constants.CENTRALITY == Constants.LOAD_CENT: 
			self.normalized_load_cent()

	#---------------------------------------------------------------------------
	def write_predictions(self, folder, sub_folder, file_name):
		if folder is not None:
			id = '0'
			for i in range(0, len(file_name)):
				if file_name[i] == ".":
					id = file_name[:i]
		name = folder + "Metadata/" + sub_folder + str(id) + "_predictions.dat"
		f = open(name, "w")
		for i in range(0, self.get_num_nodes()):
			f.write(str(self.predictions[i]) + "\n")
		f.close()

	#---------------------------------------------------------------------------
	def calculate_centralities(self, folder, sub_folder, file_name):
		file_exist = False
		id = '0'
		for i in range(0, len(file_name)):
			if file_name[i] == ".":
				id = file_name[:i]
		metadata_file = id + "_metadata.dat"
		if Constants.CENTRALITY == Constants.BETWEEN or Constants.CENTRALITY == Constants.EIGEN:
			centrality_folder = 'Betweenness/'
		elif Constants.CENTRALITY == Constants.CLOSE:
			centrality_folder = 'Closeness/'
		elif Constants.CENTRALITY == Constants.CLUSTER_COEF:
			centrality_folder = 'Cluster/'
		elif Constants.CENTRALITY == Constants.FLOW:
			centrality_folder = 'Flow/'
		elif Constants.CENTRALITY == Constants.SECOND_ORDER:
			centrality_folder = 'Second_Order/'
		elif Constants.CENTRALITY == Constants.HARMONIC:
			centrality_folder = 'Harmonic/'
		elif Constants.CENTRALITY == Constants.LOAD_CENT:
			centrality_folder = 'Load_Centrality/' 
		else:
			centrality_folder = 'Katz/' 
		sub_folder = sub_folder + centrality_folder
		file_exist = os.path.exists(folder + "Metadata/" + sub_folder + metadata_file)
		if not file_exist:
			print("No metadata found...")
			self.write_meta_data(id, folder, sub_folder)
		self.read_meta_data(folder + "Metadata/" + sub_folder + metadata_file)

	#---------------------------------------------------------------------------
	def write_meta_data(self, id, folder, sub_folder):
		if not os.path.exists(folder + "Metadata/" + sub_folder):
			os.makedirs(folder + "Metadata/" + sub_folder)
		
		name = folder + "Metadata/" + sub_folder + str(id) + "_metadata.dat"
		f = open(name, "w")

		self.normalized_centralities()
		f.write("DEGREE\n")
		for i in range(0, self.get_num_nodes()):
			f.write(str(self.get_degree(i)) + "\n")
		f.write("EIGENVECTOR\n")
		for i in range(0, self.get_num_nodes()):
			f.write(str(self.get_eigenvector_centrality(i)) + "\n")
		f.write("CENT\n")
		for i in range(0, self.get_num_nodes()):
			f.write(str(self.get_centrality(i)) + "\n")

		self.normalized_centralities_rank()
		f.write("DEGREE_RANK\n")
		for i in range(0, self.get_num_nodes()):
			f.write(str(self.get_degree(i)) + "\n")
		f.write("EIGENVECTOR_RANK\n")
		for i in range(0, self.get_num_nodes()):
			f.write(str(self.get_eigenvector_centrality(i)) + "\n")
		f.write("CENT_RANK\n")
		for i in range(0, self.get_num_nodes()):
			f.write(str(self.get_centrality(i)) + "\n")
		f.write("RANK\n")
		for i in range(0, self.get_num_nodes()):
			f.write(str(self.cent_rank[i]) + "\n")

		f.close()

	#---------------------------------------------------------------------------
	def read_meta_data(self, path):
		self.norm_degree = np.zeros(self.get_num_nodes())
		self.norm_eigen = np.zeros(self.get_num_nodes())
		self.norm_cent = np.zeros(self.get_num_nodes())
		self.cent_rank = np.zeros(self.get_num_nodes(), dtype=int)

		f = open(path, "r")
		current_vec = None
		is_int = False
		i = 0
		extra = ""
		extra_input = ""
		if Constants.TARGET == Constants.RANK:
			extra = "_RANK"
		if Constants.INPUT == Constants.RANK:
			extra_input = "_RANK"
		for line in f:
			if line == "DEGREE" + extra_input + "\n":
				current_vec = self.norm_degree
				is_int = False
				i = 0
			elif line == "EIGENVECTOR" + extra_input + "\n":
				current_vec = self.norm_eigen
				is_int = False
				i = 0
			elif line == "CENT" + extra + "\n":
				current_vec = self.norm_cent
				is_int = False
				i = 0
			elif line == "RANK" + "\n":
				current_vec = self.cent_rank
				is_int = True
				i = 0
			elif current_vec is not None and line != '-':
				if is_int:
					current_vec[i] = int(line)
				else:
					current_vec[i] = float(line)
				i += 1
				if i == self.get_num_nodes():
					current_vec = None

		if Constants.CENTRALITY == Constants.EIGEN:
			self.norm_cent =  self.norm_eigen

		f.close()

	#---------------------------------------------------------------------------
	def generate_random_graph(self, n, p):
		self.graph = nx.erdos_renyi_graph(n, p)

	#---------------------------------------------------------------------------
	def generate_smallworld_graph(self, n, p, k):
		self.graph = nx.newman_watts_strogatz_graph(n, k, p)

	#---------------------------------------------------------------------------
	def generate_scalefree_graph(self, n, m):
		self.graph = nx.barabasi_albert_graph(n, m)

	#---------------------------------------------------------------------------
	def read_file_type(self, path, delimiter):
		dot_pos = len(path) - 1
		while path[dot_pos] != '.':
			dot_pos -= 1
			if dot_pos < 0:
				print("ERROR: unkonw graph file '", path, "'")
				sys.exit()

		file_ext = path[dot_pos:]
		if file_ext == ".gexf":
			self.graph = nx.read_gexf(path, node_type=int)
		elif file_ext == ".csv" or file_ext == ".txt":
			self.graph = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int, delimiter=delimiter)
		else:
			print("ERROR: unkonw graph file '", path, "'")
			sys.exit()

		if Constants.MODE != Constants.TRAIN:
			Constants.BATCH_SIZE = self.get_num_nodes()
			print("NEW BATCH SIZE = ", Constants.BATCH_SIZE)
		#print("\n", nx.info(self.graph))

	#---------------------------------------------------------------------------
	def get_delimiter(self, folder, file):
		f = open(folder + "delimiter_info/delimiter.txt", "r")
		delimiter = '!'
		for line in f:
			words = line.split('\t')
			if words[0] == file:
				if words[1] == "space\n":
					delimiter = ' '
				elif words[1] == "tab\n":
					delimiter = '\t'
				elif words[1] == ",\n":
					delimiter = ','
				else:
					print("Unknown delimiter '" + words[1] + "' in 'delimiter.txt' file.")
					sys.exit()
		if delimiter == '!':
			print("No delimiter information found about graph '" + file + "' in the 'delimiter.txt' file.")
			sys.exit()

		return delimiter

	#---------------------------------------------------------------------------
	def save_file(self, id, folder, sub_folder):
		name = folder + sub_folder + str(id) + ".gexf"
		nx.write_gexf(self.graph, name)
		self.write_meta_data(id, folder, sub_folder)

	#---------------------------------------------------------------------------
	def read_file(self, folder, sub_folder, file_name):
		delimiter = ' '
		if Constants.MODE == Constants.REAL_NET:
			delimiter = self.get_delimiter(folder + sub_folder, file_name)
		self.read_file_type(folder + sub_folder + file_name, delimiter)
		self.calculate_centralities(folder, sub_folder, file_name)
		self.define_node_sample()
		self.predictions = np.zeros(self.get_num_nodes())
