from mpi4py import *
import sys, random
from data_lib import DataParser
from ANN import ANN

def main():
    ''' starts the distributed ANN
    '''

    # get mpi data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numNodes = comm.Get_size()

    # root node gets data from file, breaks it up, and sends out any relevant ANN information to all nodes
    if rank == 0:
	in_layer = int(sys.argv[1])
	hidden_layers = [int(sys.argv[2])]
	
	if len(sys.argv) == 6:
		hidden_layers.append(int(sys.argv[3]))
		out_layer = int(sys.argv[4])
		fileName = sys.argv[5]
	else:
		out_layer = int(sys.argv[3])
		fileName = sys.argv[4]

        # break up data (for each node and into testing/training)
	f = open(fileName)
        parser = DataParser(f)
        training_data = parser.get_training_sets(numNodes)
        test_data = parser.get_test_set()
        f.close()

        # broadcast ANN architecture to all nodes
        comm.bcast((in_layer, hidden_layers, out_layer, fileName), root=0)
	
    else:
        # receive ANN architecture from root
        in_layer, hidden_layers, out_layer, fileName = comm.bcast(None,root=0)
	training_data = None
    
    # send out/receive training data to all nodes 
    my_training_data, my_testing_data = comm.scatter(training_data, root=0)
    
    # create ANN on each node
    ann = ANN(in_layer, hidden_layers, out_layer)

    # train ANN
    random.shuffle(my_training_data)
    
    for i in range(0, len(my_training_data), 1):
        ann.train(my_training_data[i])

    # test ANN
    
    # gather results back to root node
    results = comm.gather(ann, root=0)
    
    if rank == 0:
        # combine results
        # test results
        print len(results)


if __name__ == '__main__':
    main()
