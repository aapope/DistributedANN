import unittest
from data_lib import DataParser

class TestDataParser(unittest.TestCase):
    
    def setUp(self):
        self.file = open('test/data.dat')
        self.instance = DataParser(self.file)
        
    def test_create_tuple(self):
        nums = ['1','2','3','4']
        output = self.instance._create_tuple(nums)
        self.assertEqual(output, (1.0,2.0,3.0,4.0))

    def test_parse_line(self):
        line = "1,2,3,4,5,7"
        output = self.instance._parse_line(line)
        self.assertEqual(output, ((2,3,4,5),(7,)))

    def test_parse_file(self):
        #it gets randomized
        pass

    def test_get_training_sets(self):
        sets = self.instance.get_training_sets(2)
        self.assertEqual(len(sets), 2)

    def test_divide_data(self):
        self.assertEqual(len(self.instance.test), 1)
        self.assertEqual(len(self.instance.training), 9)

    def test_separate_training_test(self):
        samples = [1,2,3,4,5,6,7,8,9,0]
        out = self.instance.separate_training_test(samples)
        self.assertEqual(([1,2,3,4,5,6,7,8], [9,0]), out)

        samples = [1,2,3,4,5,6]
        out = self.instance.separate_training_test(samples)
        self.assertEqual(([1,2,3,4], [5,6]), out)

    def test_whole(self):
        inst = DataParser(open('breast_cancer.dat'))
        print len(inst.get_training_sets(5)[1][0])

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
