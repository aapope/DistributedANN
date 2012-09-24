import random
import math

class Neuron:
    '''Set of functions for a neuron. Used to sum the input signals
    and determine the output signal
    '''

    def __init__(self):
        '''there are no inputs 
        '''
        self.inputs = 0.0
		self.last_input = 0.0

    def add_signal(self, strength):
        '''adds a signal input to the neuron (i.e. one upstream neuron fired)
        '''
        self.inputs += strength

    def fire(self):
        '''determines whether or not this neuron fires
        then resets the input accumulator.
        '''
        activation = self.activated()
		self.last_input = self.input
        self.inputs = 0
        return activation

    def activated(self):
        '''determines whether or not the neuron is 
        going to fire by comparing the signal to the
        neuron's threshold value
        '''
        return math.tanh(self.inputs)
