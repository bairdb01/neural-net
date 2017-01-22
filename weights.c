/*
 * Filename: weights.c
 * Author: Ben Baird
 * Date: Jan 20, 2017
 * Description: A list of weights for the neurons.
 * TODO: Test freeWeights
 */

 #ifndef WEIGHTS_H
 #define WEIGHTS_H
 #include "weights.h"
 #endif

Weights * initWeights() {
    Weights *weights = malloc(sizeof(Weights));
    weights->weight = 2.0 * rand()/(double)RAND_MAX  - 1.0;
    weights->next = NULL;
    return weights;
}

int addWeight(Weights *weights){
    Weights * temp = weights;
    while ( temp->next != NULL ){
        temp = temp->next;
    }
    Weights *newWeight = initWeights();
    temp->next = newWeight;
    return 0;
}

int changeWeight(Weights *weights, int index, double newWeight) {
    Weights * temp = weights;
    int i;
    for (i = 0; i < index; i++) {
        if (temp != NULL)
            temp = temp->next;
        else
            return 1;
    }

    temp->weight = newWeight;
    return 0;
}

void freeWeights (Weights *weights){
    Weights *temp = weights->next;
    while(weights != NULL){
        free(weights);
        weights = temp;
        if (temp != NULL)
            temp = temp->next;
    }
}

void printWeights(Weights *weights) {
    Weights *temp = weights;
    while (temp != NULL) {
        printf("Weight: %lf\n", temp->weight);
        temp = temp->next;
    }
}
