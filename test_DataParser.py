import unittest
from data_lib import DataParser

class TestDataParser(unittest.TestCase):
    
    def setUp(self):
        self.file = open('test/test_data.csv')
        self.instance = DataParser(self.file)
        
    def test_create_tuple(self):
        string = "1,2,3,4"
        output = self.instance.create_tuple(string)
        self.assertEqual(output, (1.0,2.0,3.0,4.0))

    def test_parse_line(self):
        line = "1,2,3,4|5,7"
        output = self.instance.parse_line(line)
        self.assertEqual(output, ((1,2,3,4),(5,7)))

    def test_parse_file(self):
        self.instance.parse_file(open('test/test_data.csv'))
        print self.instance.data
        self.assertEqual(self.instance.data, {(1,2,3,4): (5,7), 
                                              (2,3,4,7): (8,0),
                                              (3,4,6,1): (1,1)})

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
