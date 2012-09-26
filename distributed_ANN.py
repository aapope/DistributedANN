from mpi4py import *
import sys, random
from data_lib import DataParser
from ANN import ANN

EPOCHS = 10

def train(ann, vectors):
    '''Train ANN ann on vectors vectors
    '''
    random.shuffle(vectors)
    for vector in vectors:
        ann.train(vector)
    
def test(ann, vectors):
    '''Test ANN ann on vectors vectors.
    Returns the ratio of correct / total
    This test is specific to the breast cancer db right now
    '''
    correct = 0
    for vector in vectors:
        out = ann.run(vector[0])[0]
        #specific to the breast cancer db
        if (out < 0 and vector[1][0] == -1) or (out > 0 and vector[1][0] -- 1):
            correct += 1

    return float(correct)/len(vectors)

def set_up(numNodes, comm):
    '''Run by the master node, this method
    sets up the ANNs, reads in the training
    and test data sets, and scatters them to
    all the other nodes.
    '''
    try:
        in_layer = int(sys.argv[1])
        hidden_layers = [int(sys.argv[2])]
	
        if len(sys.argv) == 6:
            hidden_layers.append(int(sys.argv[3]))
            out_layer = int(sys.argv[4])
            fileName = sys.argv[5]
        else:
            out_layer = int(sys.argv[3])
            fileName = sys.argv[4]
    except:
        in_layer = 9
        hidden_layers = [1]
        out_layer = 1
        fileName = 'breast_cancer.dat'

    #break up data (for each node and into testing/training)
    f = open(fileName)
    parser = DataParser(f)
    training_data = parser.get_training_sets(numNodes)
    test_data = parser.get_test_set()
    f.close()

    # broadcast ANN architecture to all nodes
    comm.bcast((in_layer, hidden_layers, out_layer, fileName), root=0)
    return training_data, test_data, in_layer, hidden_layers, out_layer

def merge(results, in_layer, hidden_layers, out_layer):
    ann = ANN(in_layer, hidden_layers, out_layer)
    results.sort(key=lambda x: x[1], reverse=True)

    ratings = map(lambda x: x[1], results)
    i_h1_list = map(lambda i: i[0].i_h1_weights, results)
    ann.i_h1_weights = iterate_weights(ratings, i_h1_list)
    if len(hidden_layers) > 1:
        h1_h2_list = map(lambda i: i[0].h1_h2_weights, results)
        ann.h1_h2_weights = iterate_weights(ratings, h1_h2_list)
        h2_o_list = map(lambda i: i[0].h2_o_weights, results)
        ann.h2_o_weights = iterate_weights(ratings, h2_o_list)
    else:
        h1_o_list = map(lambda i: i[0].h1_o_weights, results)
        ann.h1_o_weights = iterate_weights(ratings, h1_o_list)

def iterate_weights(ratings, dic_list):
    weight_dict = {}
    for key in dic_list[0]:
        weight_dict[key] = weighted_avg(map(lambda x: x[key], dic_list), ratings)

    return weight_dict

def weighted_avg(lst, ratings):
    pass

def main():
    ''' starts the distributed ANN
    '''

    # get mpi data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numNodes = comm.Get_size()

    # root node gets data from file, breaks it up, and sends out any relevant ANN information to all nodes
    if rank == 0:
       	training_data, test_data, in_layer, hidden_layers, out_layer = set_up(numNodes, comm)
    else:
        # receive ANN architecture from root
        in_layer, hidden_layers, out_layer, fileName = comm.bcast(None,root=0)
	training_data = None
    
    # send out/receive training data to all nodes 
    my_training_data, my_testing_data = comm.scatter(training_data, root=0)
    
    # create ANN on each node
    ann = ANN(in_layer, hidden_layers, out_layer)

    # train ANN
    for i in range(EPOCHS):
        train(ann, my_training_data)
        print "Correct:", test(ann, my_testing_data), rank
    
    rating = test(ann, my_testing_data)
    
    # gather results back to root node
    results = comm.gather((ann, rating), root=0)
    
    if rank == 0:
        # combine results
        # test results
        merge(results, in_layer, hidden_layers, out_layer)


if __name__ == '__main__':
    main()
