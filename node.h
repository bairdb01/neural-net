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
    struct DataIn *data_in;    // Holds all data the node is receiving
    Weights *weights;   // List of weights to next layer
    struct Node *next;         // Pointer to next node in same layer
}Node;

typedef struct NodeLayer {
    Node *nodes;             // Linked list of nodes in this layer
    struct NodeLayer *next;  // Pointer to the next layer
}NodeLayer;

typedef struct DataIn{
    double w_val;   // Weighted value
    struct DataIn *next;
}DataIn;

// Inits a DataIn structure
DataIn *initDataIn();

// Appends a double to the receiver list
void addDataIn(double toSend, DataIn *receiver);

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
NodeLayer *initHiddenLayer(int numNodes);
