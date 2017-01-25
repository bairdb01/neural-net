/*
 * Filename: node.h
 * Author: Ben Baird
 * Date Jan 18, 2017
 * Description: Header file for nodes.c. Contains relevant functions
 *              data structures.
 */
#ifndef STDLIB_H
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

#ifndef DATA_H
#define DATA_H
#include "data.h"
#endif

typedef struct Node {
    double value;               // Nodes value after sigmoid function
    double threshold;           // Node Threshold/bias
    double err;                 // Calculated error of this node
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

// Initialize the nodes
NodeLayer * initLayer (int numNodes);

// Print a node
void printNode(Node *node);

// Print NodeLayer
void printLayer(NodeLayer *list);

// Print network
void printNetork (NodeLayer *network);

// Free DataIn structure
void freeDataIn(DataIn *dataIn);

// Free Node
void freeNode(Node* node);

// Free NodeLayer
void freeNodeLayers(NodeLayer *list);

// Iterate through all the input data of a node and calculate the weighted sum
double calcValue(Node *node);

// Sends a double (weight*value) to the start of receiver's data list
void sendData(double toSend, Node *receiver);

// Outputs the results of the nodes in the last layer
void getOutput(NodeLayer *network);

// Prints the DataIn structure
void printDataIn(DataIn *data);

