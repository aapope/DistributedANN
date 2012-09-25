from mpi4py import *
import sys
#from data_lib import DataParser

def main():
    ''' starts the distributed ANN
    '''

    # get mpi data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numNodes = comm.Get_size()

    # root node gets data from file, breaks it up, and sends out any relevant ANN information to all nodes
    if rank == 0:
	in_layer = sys.argv[1]
	hidden_layers = [sys.argv[2]]
	
	if len(sys.argv) == 6:
		hidden_layers.append(sys.argv[3])
		out_layer = sys.argv[4]
		fileName = sys.argv[5]
	else:
		out_layer = sys.argv[3]
		fileName = sys.argv[4]


        # get file
        '''
        # break up data (for each node and into testing/training)
        parser = DataParser(f)
        training_data = parser.get_training_sets(numNodes)
        test_data = parser.get_test_set()
        f.close()
	'''

        # broadcast ANN architecture to all nodes
        comm.bcast((in_layer, hidden_layers, out_layer, fileName), root=0)
    else:
        # receive ANN architecture from root
        in_layer, hidden_layers, out_layer, fileName = comm.bcast(None,root=0)
	print hidden_layers
    
    # send out/receive training data to all nodes 
    my_input = comm.scatter([(1,2), 2, 3, 4, 5, 6], root=0)
    
    # create ANN on each node
    ann = rank
    '''ann = ANN(in_neurons, hidden_neurons, out_neurons)

    # run data through ANN
    for i in range(0, len(my_input), 1):
        print ann.train(my_input[i])'''
    
    # gather results back to root node
    results = comm.gather(my_input, 0) # change to capital G if passing non-python class
    
    if rank ==0:
        # combine results
        # test results
        print results

if __name__ == '__main__':
    main()
