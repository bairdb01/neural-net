/*
 * Filename: node.h
 * Author: Ben Baird
 * Date Jan 18, 2017
 * Description: Header file for nodes.c. Contains relevant functions
 *              data structures.
 */
#ifndef STDLI_H
#define STDLIB_H
#include <stdlib.h>
#endif

#ifndef STDIO_H
#define STDIO_H
#include <stdio.h>
#endif

#ifndef MATH_H
#define MATH_H
#include <math.h>
#endif

#ifndef WEIGHTS_H
#define WEIGHTS_H
#include "weights.h"
#endif

typedef struct Node {
    double value;
    double threshold;
    double err;
    struct DataIn *data_in;     // Holds all data the node is receiving
    Weights *weights;           // List of weights to next layer
    struct Node *next;          // Pointer to next node in same layer
}Node;

typedef struct NodeLayer {
    Node *nodes;                // Linked list of nodes in this layer
    struct NodeLayer *next;     // Pointer to the next layer
    struct NodeLayer *prev;     // Pointer to the previous layer
}NodeLayer;

typedef struct DataIn{
    double w_val;   // Weighted value (weight * value)
    struct DataIn *next;
}DataIn;

// Inits a DataIn structure
DataIn * initDataIn();

// Initialize node
Node * initNode();

// Initialize the nodes in reverse order
NodeLayer * initLayer (int numNodes);

// Print a node
void printNode(Node *node);

// Print NodeLayer
void printLayer(NodeLayer *list);

// Free Node
void freeNode(Node* node);

// Free NodeLayer
void freeNodeLayer(NodeLayer *list);

// Creates a layer of hidden nodes
NodeLayer * initHiddenLayer(int numNodes);

// Iterate through all the input data of a node and calculate the weighted sum
double calcValue(Node *node);

// Sigmoid function
double sigmoid(double x);

// SigmoidError function
double sigmoidError(double x);

// Sends a double (weight*value) to the start of receiver's data list
void sendData(double toSend, Node *receiver);

// Calculates the weighted sum of the inputs of a node
double calcValue(Node *node);

// A node computes it's output values and sends to the next layer
void computeNode(Node *curNode, NodeLayer *nextLayer);

// Iterates through all nodes in a layer and sends the results to the next layer
void feedLayer(NodeLayer *layer);

// Begin feeding the network with data to receive an output
void feedNetwork();

// Outputs the results of the nodes in the last layer
void getOutput(NodeLayer *network);

// Prints the DataIn structure
void printDataIn(DataIn *data);

// Computes the error on a training pattern
// Returns the calculated error on this pattern, summed over all outputs
double backPropagateError(NodeLayer *in_layer, double *targetValues);
