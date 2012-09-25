from ANN import ANN
from data_lib import DataParser

class ANNWrapper:

    def __init__(self, in_l, hidden, ot_l, data_file):
        self.ANN = ANN(in_l, hidden, ot_l)
        self.get_data(data_file)

    def get_data(self, data_file):
        parser = DataParser(data_file)
        self.training_set = parser.get_training_sets(1)[0]
        self.test_set = parser.get_test_set()

    def train(self):
        for vector in self.training_set:
            self.ANN.train(vector)

    def test(self):
        correct = 0
        for vector in self.test_set:
            out = self.ANN.run(vector[0])
            if out == vector[1]:
                correct += 1

        print "Correct:", float(correct)/out

    def classify(self):
        pass
