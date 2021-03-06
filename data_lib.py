__author__ = "Andrew Pope"
__date__ = "23 September 2012"
__credits__ = "This module was written for CP342 Distributed Systems at Colorado College. Program by Jessa Karlberg, Andrew Pope and Cory Scott."

import random

class DataParser:
    '''This is a class for reading in a csv data file and
    getting that dataset split up. Instantiate the class
    with the data file, and call get_sets(num_sets) to get
    the data split into num_sets.
    '''

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
        self.data = []
        self._parse_file(data_file)
        self._separate_data()

    def _parse_file(self, data_file):
        '''Parses the file based on the expected file format:
        Each data point
        is on one line: the first sectionis the input vector,
        separated by commas. The second part is the correct 
        output vector, also separated by commas. The two parts
        are separated by a |. 
        e.g. '1,2,3,4|5,7\n' would be one data point, where
        (1,2,3,4) is the input vector, and (5,7) is the output
        vector.
        '''
        for d in data_file.read().split('\n')[18:]: self.data.append(self._parse_line(d))

    def _parse_line(self, line):
        '''Parses a single line of a data file
        '''
        #this is set for the breast cancer data
        tok = line.split(',')[1:]
        return (self._create_tuple(tok[:-1]), self._create_tuple(tok[-1]))

    def _create_tuple(self, nums):
        '''Creates a tuple of floats from a
        string of comma-separated numbers
        '''
        #this is set specifically for the breast cancer data
        ns = []
        if len(nums) == 1:
            if nums == '2':
                return [-1]
            else:
                return [1]
        for n in nums:
            try:
                n = float(n)
                n = -1 + (((n-1) / 9) * 2)

                ns.append(n)
            except:
                ns.append(.5)

        return tuple(ns)

    def _separate_data(self, ratio=.1):
        '''Separates data into training and
        test sets for the final ANN based
        on the ratio of test:training
        '''
        random.shuffle(self.data)
        divide = int(ratio*len(self.data))
        self.test = self.data[:divide]
        self.training = self.data[divide:]

    def get_training_sets(self, num_sets):
        '''Returns the data set inputted upon initialization
        broken up into n sets of training/test data for use
        in a classifier. The data set is an array of size
        num_sets of tuples (training_data, test_data). Each
        of the training/test data sets is an array of tuples
        (input, expected_output).
        '''
        sets = []

        chunk_size = len(self.training) / num_sets
        rem = len(self.training) % num_sets

        #The first remainder number of sets get one extra
        #data point
        for i in range(rem):
            start = max(0, (i * (chunk_size + 1)) - (chunk_size / 2))
            end = ((i + 1) * (chunk_size + 1)) + (chunk_size / 2)
            sets.append(self.separate_training_test(self.training[start:end]))

        #The rest get the rounded down amount
        for i in range(rem, num_sets):
            start = max(0, (i * chunk_size) - (chunk_size / 2))
            end = ((i + 1) * chunk_size) + (chunk_size / 2)
            sets.append(self.separate_training_test(self.training[start:end]))

        return sets

    def get_test_set(self):
        '''Returns the test set for the
        whole data set
        '''
        return self.test

    def separate_training_test(self, samples, ratio=.8):
        '''Divides the samples into two sets based
        on the ratio, a float 0 < ratio <= 1. Ratio defaults
        to .8
        '''
        div = int(ratio * len(samples))
        return (samples[:div], samples[div:])
