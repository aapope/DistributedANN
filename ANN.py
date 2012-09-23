import sys, random
from neuron import Neuron

SPARSE_THR = .15

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
			self.hlayer_1.append(Neuron())
			self.olayer.append(Neuron())
		self.hlayer_2 = None
	
	def setup_2(self):		
		self.ilayer = []
		self.hlayer_1 = []
		self.hlayer_2 = []
		self.olayer = []
		for i in range(self.in_l):
			self.ilayer.append(Neuron())
			self.hlayer_1.append(Neuron())
			self.hlayer_2.append(Neuron())
			self.olayer.append(Neuron())

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
				if new_weight > SPARSE_THRESH:
					to_return[(i,k)] = new_weight
		return to_return

	def run(input_vector):
		if len(input_vector) == len(self.ilayer):
			results = []
			for i in range(self.ilayer):
				self.ilayer[i].add_signal(input_vector[i])
				results.append(self.ilayer[i].fire())
			for i in range(self.ilayer):
				for k in range(self.hlayer1):
					if (i,k) in self.i_h1_weights:
						self.hlayer1[k].add_signal(results[i])
						
		else:
			raise Exception		
		

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
		
"""	except Exception as e:
		print e
		print 'USAGE \n python ANN.py input_nodes hidden_layer_1_nodes [hidden_layer_2_nodes] output_nodes'
"""
