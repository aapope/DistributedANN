#! /usr/bin/env python

import sys
from make_ann import ANNWrapper

if __name__ == '__main__':
    #We want the input format as follows:
    #./main.py input_layer_size hidden_layer_1 [hidden_layer_2] output_layer data_file
    try:
        in_layer = sys.argv[1]
        hidden = [sys.argv[2]]
        
        if len(sys.argv) == 6:
            hidden.append(sys.argv[3])
            out_layer = sys.argv[4]
            f = sys.argv[5]
        else:
            out_layer = sys.argv[3]
            f = sys.argv[4]

        A = ANNWrapper(in_layer, hidden, out_layer, f)
    except:
        print "Usage: main.py: python main.py input_layer_size hidden_layer_1 [hidden_layer_2] output_layer datafile"
        #testing purposes
        A = ANNWrapper(9, (3,), 1, 'breast_cancer.dat')

    A.train()
    A.test()
