/*
 * Filename: node.c
 * Author: Ben Baird
 * Date: Jan 18, 2017
 * Description: Implementation of the functions described in node.h
 *              (Functions to aid in node tasks)
 * TODO: Test free functions
 */

#ifndef NODE_H
#define NODE_H
#include "node.h"
#endif

#ifndef TIME_H
#define TIME_H
#include <time.h>
#endif

// Initialize node
Node * initNode(){
    Node *node = malloc(sizeof(Node));
    node->weights = initWeights();
    node->data_in = initDataIn();
    node->next = NULL;
    return node;
}

// Initialize the nodes
NodeLayer * initLayer (int numNodes) {
    NodeLayer *nodeLayer = malloc(sizeof(NodeLayer));
    nodeLayer->nodes = NULL;
    nodeLayer->next = NULL;

    // Create numNodes amount of nodes
    for (int i = 0; i < numNodes; i++) {
        Node *newNode = initNode();
        newNode->next = nodeLayer->nodes;
        nodeLayer->nodes = newNode;
    }

    return nodeLayer;
}

// Print a node
void printNode(Node *node) {
    printWeights(node->weights);
    printf("\n");
}

// Print NodeLayer
void printLayer(NodeLayer *list) {
    Node * curNode = list->nodes;
    while (curNode != NULL) {
        printNode(curNode);
        curNode = curNode->next;

    }
}

// Free Node
void freeNode(Node* node){
    // In nodes
    // Out Nodes
}

// Free NodeLayer
// void freeNodeLayer(NodeLayer *list) {
//     NodeLayer *temp = list->next;
//     while(list != NULL) {
//         // free(list->nodes->data_in);
//         freeWeights(list->nodes->weights);
//         freeNode(list->nodes);
//         free(list);
//         list = temp;
//         if (list != NULL)
//             temp = list->next;
//     }
// }

double sigmoid(double x) {
    return tanh(x);
}

double sigmoidError(double x) {
    return (1 - pow(x,x));
}

DataIn *initDataIn(){
    DataIn *ptr = malloc(sizeof(DataIn));
    ptr->w_val = 0.0;
    ptr->next = NULL;
    return ptr;
}

void sendData(double toSend, DataIn *receiver){
    DataIn *toAdd = initDataIn();
    toAdd->w_val = toSend;

    DataIn *temp = receiver;
    while(temp->next != NULL) {
        temp = temp->next;
    }

    temp->next = toAdd;
}

int calcValue(){
    double ans = 0.0;



    return ans;
}

// Testing
int main (void) {
    srand(time(NULL)); // Seed the random generator
    rand();

    // Create 3 input nodes
    NodeLayer *in_nodes = initLayer(3);

    // Add another weight to each node. Total = 2 per node
    addWeight(in_nodes->nodes->weights);
    addWeight(in_nodes->nodes->next->weights);
    addWeight(in_nodes->nodes->next->next->weights);

    // Change weights of first node
    changeWeight(in_nodes->nodes->weights, 0, 0.2);
    changeWeight(in_nodes->nodes->weights, 1, 0.7);

    // Change weights of second node
    changeWeight(in_nodes->nodes->next->weights, 0, -0.1);
    changeWeight(in_nodes->nodes->next->weights, 1, -1.2);

    // Change weights of third node
    changeWeight(in_nodes->nodes->next->next->weights, 0, 0.4);
    changeWeight(in_nodes->nodes->next->next->weights, 1, 1.2);
    printf("\n\t\tInput Nodes\n");
    printLayer(in_nodes);

    // Create a single hidden layer
    NodeLayer *h_layer = initLayer(2);
    in_nodes->next = h_layer;
    addWeight(h_layer->nodes->weights);
    addWeight(h_layer->nodes->next->weights);

    changeWeight(h_layer->nodes->weights, 0, 1.1);
    changeWeight(h_layer->nodes->weights, 1, 3.1);
    changeWeight(h_layer->nodes->next->weights, 0, 0.1);
    changeWeight(h_layer->nodes->next->weights, 1, 1.17);


    printf("\n\t\tHidden Layer\n");
    printLayer(h_layer);

    // Create output nodes

    NodeLayer *out_nodes = initLayer(2);
    h_layer->next = out_nodes;
    printf("\n\t\tOutput Layer\n");
    printLayer(out_nodes);

    // Input some values to nodes
    return 0;
}
