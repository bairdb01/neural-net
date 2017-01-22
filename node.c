/*
 * Filename: node.c
 * Author: Ben Baird
 * Date: Jan 18, 2017
 * Description: Implementation of the functions described in node.h
 *              (Functions to aid in node tasks)
 * Used the following website as an example for calculations:
 *         http://www.doc.ic.ac.uk/~sgc/teaching/pre2012/v231/lecture13.html
 * TODO: free functions
 * TODO: In data_in the there is an extra 0 at the end of list, doesn't affect results only memory
 * TODO: what does sigmoid (threshold  + sum do), is same as sigmoid(sum)?
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
// Free Node
void freeNode(Node *node){
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
    return (1/(1+exp(-1 * x)));
    // return tanh(x);
}

double sigmoidError(double x) {
    return x*(1 - x);
    // return (1 - (x*x));
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

void computeNode(Node *curNode, NodeLayer *nextLayer){
    // Calculate the value to send (weighted sum input passed through the sigmoid function)
    double value = calcValue(curNode);
    // printf("%lf\n", value);
    Weights *curWeight = curNode->weights;
    value = sigmoid(value);
    curNode->value = value;
    Node *nextNode = nextLayer->nodes;

    // Send the node's calculated value to each node in the next layer (value * nodeWeight)
    while (nextNode != NULL && curWeight != NULL) {
        sendData(value * curWeight->weight, nextNode);
        nextNode = nextNode->next;
        curWeight = curWeight->next;
    }
}

// Iterates through all nodes in a layer and sends the results to the next layer
void feedLayer(NodeLayer *layer) {
    Node *curNode = layer->nodes;

    // Iterate through each node in the layer
    while (curNode != NULL) {
        computeNode(curNode, layer->next);
        curNode = curNode->next;
    }
}

void feedNetwork(NodeLayer *network){
    // First layer does not need to perform a sigmoid on input, just send weight*input
    Node *in_Nodes = network->nodes;
    while (in_Nodes != NULL) {
        Weights *curWeight = in_Nodes->weights;
        Node *receiver = network->next->nodes;
        double value = in_Nodes->data_in->w_val;
        in_Nodes->value = value;

        // Send the input value to each node in the next layer (value * nodeWeight)
        while (receiver != NULL && curWeight != NULL) {
            sendData(value * curWeight->weight, receiver);
            receiver = receiver->next;
            curWeight = curWeight->next;
        }
        in_Nodes = in_Nodes->next;
    }

    // Iterate through all hidden layers and compute results
    NodeLayer *curLayer = network->next;
    while (curLayer->next != NULL) {
        feedLayer(curLayer);
        curLayer = curLayer->next;
    }

    // Output layer outputs results
    getOutput(curLayer);
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
        double value = calcValue(curNode);
        value = sigmoid(value);
        curNode->value = value;
        printf("Output Value:%lf\n", value);
        curNode = curNode->next;
    }
}

double backPropagateError(NodeLayer *in_layer, double *targetValues){
    double err = 0.0; // Error for pattern
    double learningRate = 0.3;
    double momentum = 0.6;

    // Find the output/last layer
    NodeLayer *ahead_layer = in_layer;
    while(ahead_layer->next != NULL){
        ahead_layer = ahead_layer->next;
    }

    Node *curNode = ahead_layer->nodes;
    int targetCounter = 0;
    // Iterate over all nodes in the output to calculate the pattern error and node error
    while (curNode != NULL) {
        curNode->err = sigmoidError(curNode->value) * (targetValues[targetCounter] - curNode->value);   // Error at output node i
        // Calculate the error (Desired - Output)^2
        err += (targetValues[targetCounter] - curNode->value)*(targetValues[targetCounter] - curNode->value);
        curNode = curNode->next;
        targetCounter++;
    }

    // Calculate the error relative to each err in the layer
    NodeLayer *back_layer = ahead_layer->prev;

    // Iterate over all nodes in current layer
    // TODO: Change this to go over multiple hidden layers
    curNode = back_layer->nodes;
    while (curNode != NULL) {
        curNode->err = sigmoidError(curNode->value);
        Node *iterNode = back_layer->nodes;
        Weights *curWeight = curNode->weights;
        double weightedErr = 0.0;

        // Compute weighted error in for each node in the next layer
        while (iterNode) {
            weightedErr += iterNode->err * curWeight->weight;
            iterNode = iterNode->next;
            curWeight = curWeight->next;
        }

        curNode->err *= weightedErr;
        curNode = curNode->next;
    }

    // Go through each node layer and adjust the weights based on the result in next layer
    while(back_layer != NULL) {
        curNode = back_layer->nodes;
        while (curNode != NULL) {
            Node *iterNode = ahead_layer->nodes;
            Weights *curWeight = curNode->weights;

            while(iterNode != NULL) {
                // Adjust the node weights in the back_layer
                if (momentum > 0){
                    curWeight->weight += (learningRate * curNode->value * iterNode->err) + (momentum * (curWeight->delta));
                    curWeight->delta = (learningRate * curNode->value * iterNode->err);
                } else {
                    curWeight->weight += (learningRate * curNode->value * iterNode->err);
                }
                iterNode = iterNode->next;
                curWeight = curWeight->next;
            }
            curNode = curNode->next;
        }

        // Adjust the result layer threshholds
        Node *iterNode = ahead_layer->nodes;
        while (iterNode != NULL) {
            iterNode->threshold += learningRate * iterNode->err;
            iterNode = iterNode->next;
        }

        ahead_layer = back_layer;
        back_layer = back_layer->prev;

    }
    //
    // // Adjust the Input -> Hidden layer connections
    //     // Repeat above with input layer as hidden and hidden as output

    return err;
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
    // printf("\n\t\tInput Nodes\n");
    // printLayer(in_nodes);

    // Create a single hidden layer
    NodeLayer *h_layer = initLayer(2);
    in_nodes->next = h_layer;
    h_layer->prev = in_nodes;

    addWeight(h_layer->nodes->weights);
    addWeight(h_layer->nodes->next->weights);

    changeWeight(h_layer->nodes->weights, 0, 1.1);
    changeWeight(h_layer->nodes->weights, 1, 3.1);
    changeWeight(h_layer->nodes->next->weights, 0, 0.1);
    changeWeight(h_layer->nodes->next->weights, 1, 1.17);


    // printf("\n\t\tHidden Layer\n");
    // printLayer(h_layer);

    // Create output nodes

    NodeLayer *out_nodes = initLayer(2);
    h_layer->next = out_nodes;
    out_nodes->prev = h_layer;
    // printf("\n\t\tOutput Layer\n");
    // printLayer(out_nodes);

    // Input some values to nodes
    sendData(10.0, in_nodes->nodes);
    sendData(30.0, in_nodes->nodes->next);
    sendData(20.0, in_nodes->nodes->next->next);
    // printLayer(in_nodes);

    // Begin
    feedNetwork(in_nodes);

    double *target = malloc(sizeof(double)*2);
    target[0] = 1.0;
    target[1] = 0.0;
    backPropagateError(in_nodes, target);
    printNetwork(in_nodes);
    return 0;
}
