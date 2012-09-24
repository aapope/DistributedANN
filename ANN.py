import sys, random
from neuron import Neuron

SPARSE_THR = .15
LRN_RT = .05

class ANN:
	'''This class is the neural network itself. All functions for setup,
	training, testing, and using reside here. One of these will be run
	on each node in the cluster.
	'''
	def __init__(self, in_l, hidden, ot_l, topo=None):
		'''To initialize the neural network, the input layer size,
		hidden layer size[s] (hidden layer should be a tuple of either
		one or two hidden layer sizes), the output layer size, and an
		optional topology map.
		'''
		print len(hidden), "Andrew is a dumbface"
		self.in_l = in_l
		self.ot_l = ot_l
		if len(hidden) == 1:
			self.hi_1 = hidden[0]
			self.setup_1()
		else:
			self.hi_1 = hidden[0]
			self.hi_2 = hidden[1]
			self.setup_2()
		if topo:
			print "yes"
		self.layer_set()

	def setup_1(self):
		self.ilayer = []
		self.hlayer_1 = []
		self.olayer = []
		for i in range(self.in_l):
			self.ilayer.append(Neuron())
		for i in range(self.hi_1):
			self.hlayer_1.append(Neuron())
		for i in range(self.ot_l):
			self.olayer.append(Neuron())
		self.hlayer_2 = None
	
	def setup_2(self):		
		self.ilayer = []
		self.hlayer_1 = []
		self.hlayer_2 = []
		self.olayer = []
		for i in range(self.in_l):
			self.ilayer.append(Neuron())
		for i in range(self.hi_1):
			self.hlayer_1.append(Neuron())
		for i in range(self.hi_2):
			self.hlayer_2.append(Neuron())
		for i in range(self.ot_l):
			self.olayer.append(Neuron())
		self.hlayer_2 = None

	def layer_set(self):
		self.i_h1_weights = self.set_weights(self.ilayer, self.hlayer_1)
		if self.hlayer_2 == None:
			self.h1_o_weights = self.set_weights(self.hlayer_1, self.olayer)
		else:
			self.h1_h2_weights = self.set_weights(self.hlayer_1, self.hlayer_2)
			self.h2_o_weights = self.set_weights(self.hlayer_2, self.olayer)


	def set_weights(self, layer_1, layer_2):
		to_return = {}
		for i in range(len(layer_1)):
			for k in range(len(layer_2)):
				new_weight = random.random()
				if new_weight > SPARSE_THR:
					to_return[(i,k)] = new_weight
		return to_return

	def get_results(self, in_vect, neurons_1, neurons_2, weights):
		to_return = []
		for k in range(len(neurons_2)):
			for i in range(len(neurons_1)):
				if (i, k) in weights:
					neurons_2[k].add_signal(in_vect[i]*weights[(i,k)])
			to_return.append(neurons_2[k].fire())
		return to_return
					
	def calc_error(self, given, solution):
		try:
			error = 0.0
			for i in range(max(len(given), len(solution))):
				error += (given[i] - solution[i])**(2.0)
			error = error*0.5
			return error
		except Exception as e:
			print e

	def update(self, answer, correct):
		weight_deltas = [] 		
		for i in range(len(self.olayer)):
			weight_deltas.append((1.0 - answer[i])*(answer[i])*(correct[i] - answer[i]))
		pass

	def train(self, data_tup):
		data, soln = data_tup
		answer = self.run(data)
		update(answer, soln)

	def run(self, input_vector):
		results = []
		for i in range(len(self.ilayer)):
			self.ilayer[i].add_signal(input_vector[i])
			results.append(self.ilayer[i].fire())
		results = self.get_results(results, self.ilayer, self.hlayer_1, self.i_h1_weights)
		last = self.hlayer_1
		if self.hlayer_2 != None:		
			results = self.get_results(results, self.hlayer_1, self.hlayer_2, self.h1_h2_weights)
			last = self.hlayer_2
			last_weights = self.h2_o_weights
		else:
			last_weights = self.h1_o_weights
		results = self.get_results(results, last, self.olayer, last_weights)
		return results	

if __name__ == '__main__':
	print sys.argv
	args = map(lambda x: int(x), sys.argv[1:])
	in_layer = args[0]
	out_layer = args[-1]
	hi_1 = args[1]
	if len(args) == 3:
		my_ann = ANN(in_layer, (hi_1,), out_layer)
	elif len(args) == 4:
		hi_2 = args[2]
		my_ann = ANN(in_layer, (hi_1, hi_2), out_layer)
	print len(my_ann.ilayer), len(my_ann.hlayer_1), len(my_ann.olayer) 
	print my_ann.run([0,1,1,0,1])
		
"""	except Exception as e:
		print e
		print 'USAGE \n python ANN.py input_nodes hidden_layer_1_nodes [hidden_layer_2_nodes] output_nodes'
"""
