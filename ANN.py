import sys, random


SPARSE_THR = .15

class Neuron:
	def __init__(self):
		self.out = 1

class ANN:
	def __init__(self, in_l, hidden, ot_l, topo):
		print len(hidden)
		self.in_l = in_l
		self.ot_l = ot_l
		if len(hidden) == 1:
			self.hi_1 = hidden[0]
			self.setup_1()
		else:
			self.hi_1 = hidden[0]
			self.hi_2 = hidden[1]
			self.setup_2()
		self.layer_set()

	def __str__(self):
		return "Andrew is a dumb face."
	
	def setup_1(self):
		self.ilayer = []
		for i in range(self.in_l):
			self.ilayer.append(Neuron())
		self.hlayer_1 = []
		for i in range(self.hi_1):
			self.hlayer_1.append(Neuron())
		self.hlayer_2 = None
		self.olayer = []
		for i in range(self.ot_l):
			self.olayer.append(Neuron())
	
	def setup_2(self):		
		self.ilayer = []
		for i in range(self.in_l):
			self.ilayer.append(Neuron())
		self.hlayer_1 = []
		for i in range(self.hi_1):
			self.hlayer_1.append(Neuron())
		self.hlayer_2 = []
		for i in range(self.hi_1):
			self.hlayer_2.append(Neuron())
		self.olayer = []
		for i in range(self.ot_l):
			self.olayer.append(Neuron())

	def layer_set(self):
		self.i_h1_weights = self.set_weights(self.ilayer, self.hlayer_1)
		if self.hlayer_2 == None:
			self.h1_o_weights = self.set_weights(self.hlayer_1, self.olayer)
		else:
			self.h1_h2_weights = self.set_weights(self.hlayer_1, self.hlayer_2)
			self.h2_o_weights = self.set_weights(self.hlayer_2, self.olayer)


	def set_weights(self, layer_1, layer_2):
		to_return = []
		for i in range(len(layer_1)):
			for k in range(len(layer_2)):
				to_return.append((i,k,random.random()))
		return to_return
		

if __name__ == '__main__':
	print sys.argv
	args = map(lambda x: int(x), sys.argv[1:])
	in_layer = args[0]
	out_layer = args[-1]
	hi_1 = args[1]
	if len(args) == 3:
		my_ann = ANN(in_layer, (hi_1,), out_layer, None)
	elif len(args) == 4:
		hi_2 = args[2]
		my_ann = ANN(in_layer, (hi_1, hi_2), out_layer, None)
	print my_ann
		
"""	except Exception as e:
		print e
		print 'USAGE \n python ANN.py input_nodes hidden_layer_1_nodes [hidden_layer_2_nodes] output_nodes'
"""
