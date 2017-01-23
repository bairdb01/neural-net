/*
 * Filename: node.c
 * Author: Ben Baird
 * Date: Jan 18, 2017
 * Description: Implementation of the functions described in node.h
 *              (Functions to aid in node tasks)
 * Used the following website as an example for calculations:
 *         http://www.doc.ic.ac.uk/~sgc/teaching/pre2012/v231/lecture13.html
 * TODO: free functions
 * TODO: Test multiple hidden layers
 * TODO: Save network
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

void feedNetwork(NodeLayer *network, double *input, int nInputs){
    // Read the input data into the input layer
    Node *receiver = network->nodes;
    for (int i = 0; i < nInputs; i++) {
        sendData(input[i], receiver);
        receiver = receiver->next;
    }
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

double backPropagateError(NodeLayer *in_layer, double *targetValues, double learningRate, double momentum){
    double err = 0.0; // Error for pattern

    // Find the output/last layer
    NodeLayer *ahead_layer = in_layer;
    while(ahead_layer->next != NULL){
        ahead_layer = ahead_layer->next;
    }

    Node *curNode = ahead_layer->nodes;
    int targetCounter = 0;
    // Compute error at output layer
    while (curNode != NULL) {
        curNode->err = sigmoidError(curNode->value) * (targetValues[targetCounter] - curNode->value);   // Error at output node i
        // Calculate the total error (Desired - Output)^2
        err += (targetValues[targetCounter] - curNode->value)*(targetValues[targetCounter] - curNode->value);
        curNode = curNode->next;
        targetCounter++;
    }

    // Calculate the error relative to each err in the layer
    NodeLayer *back_layer = ahead_layer->prev;

    // Iterate over all nodes in current layer
    // TODO: Change this to go over multiple hidden layers?
    curNode = back_layer->nodes;

    while (curNode != NULL) {
        curNode->err = sigmoidError(curNode->value);
        Node *iterNode = ahead_layer->nodes;
        Weights *curWeight = curNode->weights;
        double weightedErr = 0.0;

        // Compute weighted error in for each node in the next layer
        while (iterNode != NULL) {
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
    return err;
}

// double trainNetwork(FILE *trainFP, double learningRate, double momentum, int rows) {
double trainNetwork(DataSet *set, NodeLayer *network, double learningRate, double momentum){
    int epoch = 0;
    double err = 999999;    // Cumulative error over an epoch
    int maxIter = 10000;
    double minErr = 0.00001;

    while (epoch < maxIter && err > minErr) {
        err = 0;
        int dataLeft = set->nData;  // How much of the dataset is left
        int *work = malloc(sizeof(int)*set->nData);

        // Put all data id's into the work list
        Data *iter = set->data;

        for (int i = 0 ; i < set->nData; i++) {
            work[i] = iter->id;
            iter = iter->next;
        }

        while (dataLeft != 0) {
            Data *pattern = set->data;

            // Choose a random pattern from the working set
            int used = 1;
            int index;
            while (used) {
                index = rand() % (dataLeft);
                int workable = -1;
                for (int i = 0; i < set->nData; i++) {
                    if (work[i] != -1) {
                        workable++;
                    }
                    if (workable == index) {
                        used = 0;
                        break;
                    }
                }
            }
            while(pattern->id != index && pattern != NULL)
                pattern = pattern->next;

            // Remove pattern from working set
            work[index] = -1;

            feedNetwork(network, pattern->inputs, set->nIn);
            err += backPropagateError(network, pattern->targets, learningRate, momentum);
            dataLeft--;
        }
        free(work);
        // Scale error by number of samples
        err /= (set->nData * set->nTargets);
        epoch++;
    }
    return err;
}

NodeLayer *createNetwork(FILE *netFP) {
    NodeLayer *network = NULL;
    int numNodes = 0;
    int nPrev = -1;
    // Read the network file for specifications
    while (fscanf(netFP, "%d %*s", &numNodes) > 0) {
        NodeLayer *toAdd = initLayer(numNodes);

        if (network == NULL)
            network = toAdd;
        else {
            NodeLayer *iter = network;
            while(iter->next != NULL)
                iter = iter->next;
            iter->next = toAdd;
            toAdd->prev = iter;
        }

        // Create weight data structure for each node
        if (nPrev > 0) {
            Node *iter = toAdd->prev->nodes;
            while (iter != NULL) {
                for (int i = 0; i < (nPrev-1); i++) {
                    Weights *newWeight = initWeights();
                    newWeight->next = iter->weights;
                    iter->weights = newWeight;
                }
                iter = iter->next;
            }
        }
        nPrev = numNodes;
    }

    return network;
}

DataSet * loadData(FILE *netFP, FILE *trainFP) {
    int nTargets = 0;
    int nInputs = 0;
    fscanf(netFP, "%d %*s", &nInputs);

    // Get number of output nodes (last number read)
    while (fscanf(netFP, "%d %*s", &nTargets) > 0);

    DataSet *set = initDataSet (nInputs, nTargets);
    char *line = malloc(sizeof(char)*501);
    fgets(line, 500, trainFP); // Skip the first line containing column names

    // Load data into data structures
    while (fgets(line, 500, trainFP) != NULL) {
        char *token = strtok(line, " ");
        Data *data = initData(nInputs, nTargets, set->nData);
        int inCount = 0;
        int tarCount = 0;

        while (token != NULL) {
            double num = 0.0;
            sscanf(token, "%lf", &num);
            if (inCount < nInputs) {
                data->inputs[inCount] = num;
                inCount++;
            } else if (tarCount < nTargets){
                data->targets[tarCount] = num;
                tarCount++;
            }
            token = strtok(NULL, " ");
        }
        data->next = set->data;
        set->data = data;
        set->nData++;
    }
    return set;
}

int main (void) {
    srand(time(NULL)); // Seed the random generator
    rand();
    double learningRate = 0.0;
    double momentum = 0.0;
    char *netStr = malloc(sizeof(char)*100);
    netStr = strcpy(netStr, "network.txt");

    FILE *netFP = fopen(netStr, "rw");
    if (netFP == NULL) {
        printf("Could not find network.txt file\n");
        return 1;
    }

    printf("Enter learning rate:\n");
    scanf("%lf", &learningRate);
    printf("Enter momentum:\n");
    scanf("%lf", &momentum);

    char *trainLoc = malloc(sizeof(char)*100);
    // printf("Where is the training file located?\n");
    // scanf("%s", trainLoc);

    // FILE *trainFP = fopen(trainLoc, "r");
    FILE *trainFP = fopen("training/iris.txt", "r");
    if (trainFP == NULL) {
        printf("Invalid file location\n");
        return 1;
    }

    // Create network
    NodeLayer *network = createNetwork(netFP);

    // Read the datafiles to memory
    rewind(netFP);
    DataSet *trainSet = loadData(netFP, trainFP);

    FILE *testFP = fopen("test/iris.txt", "r");
    if (testFP == NULL) {
        printf("Invalid test file location\n");
        return 1;
    }
    rewind(netFP);
    DataSet *testSet = loadData(netFP, testFP);

    // Begin training
    trainNetwork(trainSet, network, learningRate, momentum);

    // Test the network
    printf("\t\t\tTesting\n");
    Data *iter = testSet->data;
    while (iter != NULL) {
        printf("\nInput: %lf %lf %lf %lf\n", iter->inputs[0], iter->inputs[1], iter->inputs[2], iter->inputs[3]);
        printf("Target: %lf %lf %lf\n", iter->targets[0], iter->targets[1], iter->targets[2]);
        feedNetwork(network, iter->inputs, 4);
        iter = iter->next;
    }

    //Save network weights, deltas, learningRate, momentum, etc

    free(trainLoc);
    fclose(netFP);
    fclose(trainFP);

    // // Create 4 input nodes
    //
    // NodeLayer *in_nodes = initLayer(4);
    //
    // // Add another weight to each node. Total = 3 per node
    // addWeight(in_nodes->nodes->weights);
    // addWeight(in_nodes->nodes->weights);
    // addWeight(in_nodes->nodes->next->weights);
    // addWeight(in_nodes->nodes->next->weights);
    // addWeight(in_nodes->nodes->next->next->weights);
    // addWeight(in_nodes->nodes->next->next->weights);
    // addWeight(in_nodes->nodes->next->next->next->weights);
    // addWeight(in_nodes->nodes->next->next->next->weights);
    //
    // // Change weights of first node
    // // changeWeight(in_nodes->nodes->weights, 0, 0.2);
    // // changeWeight(in_nodes->nodes->weights, 1, 0.7);
    //
    // // Change weights of second node
    // // changeWeight(in_nodes->nodes->next->weights, 0, -0.1);
    // // changeWeight(in_nodes->nodes->next->weights, 1, -1.2);
    //
    // // Change weights of third node
    // // changeWeight(in_nodes->nodes->next->next->weights, 0, 0.4);
    // // changeWeight(in_nodes->nodes->next->next->weights, 1, 1.2);
    // // printf("\n\t\tInput Nodes\n");
    // // printLayer(in_nodes);
    //
    // // Create a single hidden layer
    // NodeLayer *h_layer = initLayer(3);
    // in_nodes->next = h_layer;
    // h_layer->prev = in_nodes;
    //
    // addWeight(h_layer->nodes->weights);
    // addWeight(h_layer->nodes->weights);
    // addWeight(h_layer->nodes->next->weights);
    // addWeight(h_layer->nodes->next->weights);
    // addWeight(h_layer->nodes->next->next->weights);
    // addWeight(h_layer->nodes->next->next->weights);
    //
    // // changeWeight(h_layer->nodes->weights, 0, 1.1);
    // // changeWeight(h_layer->nodes->weights, 1, 3.1);
    // // changeWeight(h_layer->nodes->next->weights, 0, 0.1);
    // // changeWeight(h_layer->nodes->next->weights, 1, 1.17);
    //
    //
    // // printf("\n\t\tHidden Layer\n");
    // // printLayer(h_layer);
    //
    // // Create output nodes
    // NodeLayer *out_nodes = initLayer(3);
    // addWeight(out_nodes->nodes->weights);
    // addWeight(out_nodes->nodes->weights);
    // addWeight(out_nodes->nodes->next->weights);
    // addWeight(out_nodes->nodes->next->weights);
    // addWeight(out_nodes->nodes->next->next->weights);
    // addWeight(out_nodes->nodes->next->next->weights);
    // h_layer->next = out_nodes;
    // out_nodes->prev = h_layer;
    // // printf("\n\t\tOutput Layer\n");
    // // printLayer(out_nodes);
    //
    // // Input some values to nodes
    // // sendData(5.9, in_nodes->nodes);
    // // sendData(3.0, in_nodes->nodes->next);
    // // sendData(5.1, in_nodes->nodes->next->next);
    // // sendData(1.8, in_nodes->nodes->next->next->next);
    // // printLayer(in_nodes);
    //
    // // Data *testInput = initData(4, 3, 0);
    // // testInput->next = initData(4,3,1);
    // // testInput->next->next = initData(4,3,2);
    // // testInput->next->next->next = initData(4,3,3);
    // double *testInputs = malloc(sizeof(double)*4);
    // testInputs[0] = 5.9;
    // testInputs[1] = 3.0;
    // testInputs[2] = 5.1;
    // testInputs[3] = 1.8;
    //
    // // Begin
    // feedNetwork(in_nodes, testInputs, 4);
    //
    // double *target = malloc(sizeof(double)*3);
    // target[0] = 0.0;
    // target[1] = 0.0;
    // target[2] = 1.0;
    // // double learningRate = 0.3;
    // // double momentum = 0.6;
    //
    // backPropagateError(in_nodes, target, learningRate, momentum);
    // printNetwork(in_nodes);

    return 0;
}
