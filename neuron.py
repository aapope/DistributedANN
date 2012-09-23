import random

class Neuron:
    '''Set of functions for a neuron. Used to sum the input signals
    and determine the output signal
    '''

    def __init__(self):
        '''there are no inputs 
        '''
        self.inputs = 0
        self.threshold = random.randrange(1,2)

    def add_signal(self, strength):
        '''adds a signal input to the neuron (i.e. one upstream neuron fired)
        '''
        self.inputs += strength

    def fire(self):
        '''determines whether or not this neuron fires
        then resets the input accumulator.
        '''
        if self.activated():
            ret = True
        else:
            ret = False
        self.inputs = 0
        return ret

    def activated(self):
        '''determines whether or not the neuron is 
        going to fire by comparing the signal to the
        neuron's threshold value
        '''
        if self.signal > self.threshold:
            return True
