from ANN import ANN
from data_lib import DataParser

class ANNWrapper:

    def __init__(self, in_l, hidden, ot_l, data_file):
        self.ANN = ANN(in_l, hidden, ot_l)
        self.get_data(open(data_file))

    def get_data(self, data_file):
        parser = DataParser(data_file)
        sets = parser.get_training_sets(1)
        #one set, so take the first of the trainings sets
        #and take the first of that (i.e. not the test set)
        self.training_set = sets[0][0]
        self.test_set = parser.get_test_set()

    def train(self):
        print len(self.training_set)
        for vector in self.training_set:
            self.ANN.train(vector)

    def test(self):
        correct = 0
        for vector in self.test_set:
            out = self.ANN.run(vector[0])[0]
            print out, vector[1]
            #check which it's closest to for this instead
            if (out < 0 and vector[1][0] == -0.2) or (out > 0 and vector[1][0] == -0.6):
                correct += 1

        print "Correct:", float(correct)/len(self.test_set)

    def classify(self):
        pass
