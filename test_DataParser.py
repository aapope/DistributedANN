import unittest
from data_lib import DataParser

class TestDataParser(unittest.TestCase):
    
    def setUp(self):
        self.file = open('test/test_data.csv')
        self.instance = DataParser(self.file)
        
    def test_create_tuple(self):
        string = "1,2,3,4"
        output = self.instance._create_tuple(string)
        self.assertEqual(output, (1.0,2.0,3.0,4.0))

    def test_parse_line(self):
        line = "1,2,3,4|5,7"
        output = self.instance._parse_line(line)
        self.assertEqual(output, ((1,2,3,4),(5,7)))

    def test_parse_file(self):
        #it gets randomized
        pass

    def test_get_training_sets(self):
        print self.instance.get_training_sets(2)

    def test_separate_training_test(self):
        samples = [1,2,3,4,5]
        out = self.instance.separate_training_test(samples)
        self.assertEqual(([1,2,3,4], [5]), out)

        samples = [1,2,3,4,5,6]
        out = self.instance.separate_training_test(samples)
        self.assertEqual(([1,2,3,4], [5,6]), out)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
