/*
 * Filename: node.c
 * Author: Ben Baird
 * Date: Jan 18, 2017
 * Description: Implementation of the functions described in node.h
 *              (Functions to aid in node tasks)
 * Used the following website as an example for calculations:
 *         http://www.doc.ic.ac.uk/~sgc/teaching/pre2012/v231/lecture13.html
 * TODO: Test multiple hidden layers
 * TODO: In data_in the there is an extra 0 at the end of list, doesn't affect results only memory
 */

#ifndef NODE_H
#define NODE_H
#include "node.h"
#endif

#ifndef TIME_H
#define TIME_H
#include <time.h>
#endif

#ifndef STRING_H
#define STRING_H
#include <string.h>
#endif

// Initialize node
Node * initNode(){
    Node *node = malloc(sizeof(Node));
    node->value = 0.0;
    node->threshold = rand()/(double)RAND_MAX;
    node->err = 0.0;
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
    nodeLayer->prev = NULL;
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
    printf("Value: %lf\n", node->value);
    printf("Threshold: %lf\n", node->threshold);
    printf("Error: %lf \n",node->err);
    printDataIn(node->data_in);
    printWeights(node->weights);
    printf("\n");
}

void printDataIn(DataIn *data) {
    DataIn *temp = data;
    printf("In: ");
    while (temp != NULL) {
        printf("%lf ", temp->w_val);
        temp = temp->next;
    }
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

void printNetwork(NodeLayer *network) {
    NodeLayer *temp = network;
    int i = 0;
    while (temp != NULL) {
        printf("\t\t--- Layer %d ----\n", i);
        printLayer(temp);
        i++;
        temp = temp->next;
    }

}

void freeDataIn(DataIn *dataIn){
    DataIn *temp;
    while (dataIn != NULL){
        temp = dataIn;
        dataIn = dataIn->next;
        free(temp);
    }
}

void freeNode(Node *node){
    freeDataIn(node->data_in);
    freeWeights(node->weights);
    free(node);
}

void freeNodeLayers(NodeLayer *list) {
    while(list != NULL) {
        Node *curNode = list->nodes;
        while (curNode != NULL) {
            Node *next = curNode->next;
            freeNode(curNode);
            curNode = next;
        }
        NodeLayer *prev = list;
        list = list->next;
        free(prev);
    }
}

DataIn *initDataIn(){
    DataIn *ptr = malloc(sizeof(DataIn));
    ptr->w_val = 0.0;
    ptr->next = NULL;
    return ptr;
}

void sendData(double toSend, Node *receiver){
    DataIn *toAdd = initDataIn();
    toAdd->w_val = toSend;
    toAdd->next = receiver->data_in;
    receiver->data_in = toAdd;
}

double calcValue(Node *node){
    double ans = 0.0;

    DataIn *temp = node->data_in;
    DataIn *rm = temp;
    while (temp != NULL){
        ans += temp->w_val;
        temp = temp->next;
        free(rm);
        rm = temp;
    }
    node->data_in = initDataIn();
    return ans;
}

void getOutput(NodeLayer *network){
    NodeLayer *curLayer = network;
    while(curLayer->next != NULL){
        curLayer = curLayer->next;
    }

    Node *curNode = curLayer->nodes;

    // Print the node's calculated value
    printf("\n\t\tOutput Values\n");
    while (curNode != NULL) {
        printf("Output Value:%lf\n", curNode->value);
        curNode = curNode->next;
    }
}
