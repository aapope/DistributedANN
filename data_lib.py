
class DataParser:

    def __init__(self, data_file):
        '''Takes the csv file in which the data are stored.
        Expected file format is as follows. Each data point
        is on one line: the first sectionis the input vector,
        separated by commas. The second part is the correct 
        output vector, also separated by commas. The two parts
        are separated by a |. 
        e.g. '1,2,3,4|5,7\n' would be one data point, where
        (1,2,3,4) is the input vector, and (5,7) is the output
        vector.
        '''
        self.data = {}
        self.parse_file(data_file)

    def parse_file(self, data_file):
        for d in data_file.read().split('\n'): self.parse_line(d)

    def parse_line(self, line):
        in_p, out_p = line.split('|')
        self.data[self.create_tuple(in_p)] = self.create_tuple(out_p)
        return (self.create_tuple(in_p), self.create_tuple(out_p))

    def create_tuple(self, string):
        return tuple(map(lambda x: float(x), string.split(',')))

    def get_sets(self, num_sets):
        '''Returns the data set inputted upon initialization
        broken up into n sets of training/test data for use
        on a classifier. 
        '''
        pass
