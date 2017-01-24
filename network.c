/*
 * Filename:
 * Author: Ben Baird
 * Date: Jan 23, 2017
 * Description: Implementation for a multi-layer perceptron with backpropagation
 */

#ifndef NODE_H
#define NODE_H
#include "node.h"
#endif

#ifndef DATA_H
#define DATA_H
#include "data.h"
#endif

#ifndef TIME_H
#define TIME_H
#include <time.h>
#endif

#ifndef STRING_H
#define STRING_H
#include <string.h>
#endif

double sigmoid(double x) {
    return (1/(1+exp(-1 * x)));
    // return tanh(x);
}

double sigmoidError(double x) {
    return x*(1 - x);
    // return (1 - (x*x));
}

void computeNode(Node *curNode, NodeLayer *nextLayer){
    // Calculate the value to send (weighted sum input passed through the sigmoid function)
    double value = calcValue(curNode);
    Weights *curWeight = curNode->weights;
//    value = sigmoid(value);
     value = sigmoid(curNode->threshold + value);
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

    // Calculate the output values
    while(curLayer->next != NULL){
        curLayer = curLayer->next;
    }

    Node *curNode = curLayer->nodes;

    // Print the node's calculated value
    while (curNode != NULL) {
        double value = calcValue(curNode);
//        value = sigmoid(value);
         value = sigmoid(curNode->threshold + value);
        curNode->value = value;
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
    // TODO: Change this to go over multiple hidden layers
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

    // Go through each node layer and adjust the weights based on the result in the next layer
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
    int maxIter = 1;
    double minErr = 0.00001;

    printf("Training...");
    fflush(stdout);
    while (epoch < maxIter && err > minErr) {
        if (epoch % 10000 == 0){
            printf(".");
            fflush(stdout);
        }
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
    printf("\n");
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
                for (int i = 0; i < (numNodes-1); i++) {
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
    free(line);
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
    printf("Creating network...\n");
    NodeLayer *network = createNetwork(netFP);

    // Read the datafiles to memory
    printf("Loading training data into memory...\n");
    rewind(netFP);
    DataSet *trainSet = loadData(netFP, trainFP);

    char *testLoc = malloc(sizeof(char)*100);
    // printf("Where is the test file located?\n");
    // scanf("%s", testLoc);
    // FILE *testFP = fopen(testLoc, "r");
    FILE *testFP = fopen("test/iris.txt", "r");
    if (testFP == NULL) {
        printf("Invalid test file location\n");
        return 1;
    }

    rewind(netFP);
    printf("Loading test data into memory...\n");
    DataSet *testSet = loadData(netFP, testFP);
    
    // Begin training
    // printNetwork(network);
     trainNetwork(trainSet, network, learningRate, momentum);

    // Get the output layer
    NodeLayer *out_layer = network;
    while (out_layer->next != NULL)
        out_layer = out_layer->next;

    // Test the network
    printf("\n\t\tTesting\n");
    Data *iter = testSet->data;
    while (iter != NULL) {
        printf("\nInput: %lf %lf %lf %lf\n", iter->inputs[0], iter->inputs[1], iter->inputs[2], iter->inputs[3]);
        printf("Target: %lf %lf %lf\n", iter->targets[0], iter->targets[1], iter->targets[2]);
        feedNetwork(network, iter->inputs, 4);
        getOutput(out_layer);
        iter = iter->next;
    }

    // Need to save network weights, deltas, learningRate, momentum, etc
    fclose(netFP);
    fclose(trainFP);
    fclose(testFP);
    free(testLoc);
    free(trainLoc);
    free(netStr);
    freeDataSet(testSet);
    freeDataSet(trainSet);
    freeNodeLayers(network);
    return 0;
}
