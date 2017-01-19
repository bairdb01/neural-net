/*
 * Filename: node.c
 * Author: Ben Baird
 * Date: Jan 18, 2017
 * Description: Implementation of the functions described in node.h
 *              (Functions to aid in node tasks)
 */
#include <stdio.h>
#include <stdlib.h>
#include "node.h"

// Initialize node
Node * initNode(){
    Node *node = malloc(sizeof(Node));
    node->weight = 1;
    node->inNodes = malloc(sizeof(Node));
    node->outNodes = malloc(sizeof(Node));
    return node;
}

// Initialize the nodes
Node * initNodeList (int numNodes) {
    NodeList *nodeList = NULL;
    NodeList *curNode = NULL;

    // Create numNodes amount of nodes
    for (int i = 0; i < numNodes; i++) {
        curNode = malloc(sizeof(NodeList));
        curNode->node = initNode();
        curNode->next = nodeList;
        nodeList = curNode;
    }
}

// Print a node
void printNode(Node *node) {
    printf("Weight: %lf\n", node->weight);
    // inNodes
    // outNodes
}
// Print NodeList
void printNodeList(NodeList *list) {
    NodeList * curNode = list;
    while (curNode != NULL) {
        printNode(curNode->node);
        curNode = curNode->next;
    }
}

// Free Node
void freeNode(Node* node){
    // In nodes
    // Out Nodes
}

// Free NodeList
void freeNodeList(NodeList *list) {
    NodeList *temp = list->next;
    while(list != NULL) {
        freeNode(list->node);
        free(list);
        list = temp;
        if (list != NULL)
            temp = list->next;
    }
}

// Testing
int main (void) {
    NodeList *in_nodes;

    // Input some values to nodes

    // Create input nodes

    // Create hidden layer

    // Create output nodes

    return 0;
}
