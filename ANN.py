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

	#two different setup methods, depending on the number of hidden layers'''
	def setup_1(self):
		self.ilayer = []
		self.hlayer_1 = []
		self.olayer = []
		#for now, just append random neurons.
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

	#depending on # of hidden nodes, initialize the weight dicts in different ways
	def layer_set(self):
		self.i_h1_weights = self.init_weights(self.ilayer, self.hlayer_1)
		if self.hlayer_2 == None:
			self.h1_o_weights = self.init_weights(self.hlayer_1, self.olayer)
		else:
			self.h1_h2_weights = self.init_weights(self.hlayer_1, self.hlayer_2)
			self.h2_o_weights = self.init_weights(self.hlayer_2, self.olayer)

	#given two layers, create lots of random weights. 
	def init_weights(self, layer_1, layer_2):
		to_return = {}
		for i in range(len(layer_1)):
			for k in range(len(layer_2)):
				'''SPARSE_THR controls the sparseness of the network; 
				connections whose weight is lower than this value are not created.'''
				new_weight = random.random()
				if new_weight > SPARSE_THR:
					to_return[(i,k)] = new_weight
		return to_return

	'''given two layers of perceptrons, calculate the signal passed to each perceptron in the second layer'''
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

	'''update function. Take in a given answer and the "correct" version, calculate the error,
	and then backpropagate it through the network.'''
	def update(self, answer, correct):
		for i in range(len(self.olayer)):
			inp = answer[i]
			self.olayer[i].delta = (inp)*(1.0-inp)*(correct[i] - inp)
		if self.hlayer_2:
			self.deltify(self.hlayer_2, self.olayer, self.h2_o_weights)
			self.deltify(self.hlayer_1, self.hlayer_2, self.h1_h2_weights)
		else:
			self.deltify(self.hlayer_1, self.olayer, self.h1_o_weights)			
		self.deltify(self.ilayer, self.hlayer_1, self.i_h1_weights)
		self.set_weights()

	'''simply calls apply_deltas for each present layer.'''
	def set_weights(self):
		if self.hlayer_2:
			self.apply_deltas(self.hlayer_2, self.olayer, self.h2_o_weights)
			self.apply_deltas(self.hlayer_1, self.hlayer_2, self.h1_h2_weights)
		else:
			self.apply_deltas(self.hlayer_1, self.olayer, self.h1_o_weights)
		self.apply_deltas(self.ilayer, self.hlayer_1, self.i_h1_weights)
			
	'''after all of the deltas have been calculated and stored, apply them to each node.'''
	def apply_deltas(self, layer1, layer2, weights):
		for i in range(len(layer1)):
			for k in range(len(layer2)):
				if (i,k) in weights:
					weights[(i,k)] = weights[(i,k)] + LRN_RT*layer2[k].last_output*layer2[k].delta

	'''calculate the delta for a given node, and then store it.'''
	def deltify(self, first_layer, second_layer, weights):
		for i in range(len(first_layer)):
			summed = 0.0
			inp = first_layer[i].last_output
			for k in range(len(second_layer)):
				if (i,k) in weights:
					summed += weights[(i,k)]*second_layer[k].delta
			first_layer[i].delta = (inp)*(1.0-inp)*summed		

	'''take in a vector and its solution, and then send those to several training submethods.'''
	def train(self, data_tup):
		data, soln = data_tup
		answer = self.run(data)
		self.update(answer, soln)

	'''get the output from some particular vector.'''
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
	ans = []
	for i in range(out_layer):
		ans.append(random.random())
	print ans, '\n*****'
	tester = 1
	for i in range(100000000):
		my_ann.train(([0,1,1,0,1], ans))
		if i % tester == 0:
			print my_ann.run([0,1,1,0,1])
			tester *= 10		
"""	except Exception as e:
		print e
		print 'USAGE \n python ANN.py input_nodes hidden_layer_1_nodes [hidden_layer_2_nodes] output_nodes'
"""
