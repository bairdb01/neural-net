/*
 * Filename: network.h
 * Author: Ben Baird
 * Date: Jan 23, 2017
 * Description: Header file for network.c
 */

 // Sigmoid function
 double sigmoid(double x);

 // SigmoidError function
 double sigmoidError(double x);

 // Calculates the weighted sum of the inputs of a node
 double calcValue(Node *node);

 // A node computes it's output values and sends to the next layer
 void computeNode(Node *curNode, NodeLayer *nextLayer);

 // Iterates through all nodes in a layer and sends the results to the next layer
 void feedLayer(NodeLayer *layer);

 // Begin feeding the network with data to receive an output
 void feedNetwork(NodeLayer *network, double *input, int nInputs);

 // Computes the error on a training pattern
 // Returns the calculated error on the previously executed pattern, summed over all outputs
 double backPropagateError(NodeLayer *in_layer, double *targetValues, double learningRate, double momentum);

 // Trains a network
 double trainNetwork(DataSet *set, NodeLayer *network, double learningRate, double momentum);

 // Creates a network with params being the number of nodes per layer (%d %d %d etc)
 NodeLayer *createNetwork(FILE *netFP);
