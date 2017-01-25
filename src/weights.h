/*
 * Filename: weights.h
 * Author: Ben Baird
 * Date: Jan 20, 2017
 * Description: Function headers and structs for weights.c
 */
#ifndef STDLI_H
#define STDLIB_H
#include <stdlib.h>
#endif

#ifndef STDIO_H
#define STDIO_H
#include <stdio.h>
#endif

#ifndef TIME_H
#define TIME_H
#include <time.h>
#endif

 typedef struct Weights {
     double weight;
     double delta;
     struct Weights * next;
 }Weights;

// Allocates memory and initializes a Weights variable with a random weight
Weights * initWeights();

// Add a Weights variable to the Weights list
int addWeight(Weights *weights);

// Change the weight of one of the indexed weights to newWeight
int changeWeight(Weights *weights, int index, double newWeight);

// Frees all memory allocated for weights
void freeWeights (Weights *weights);

void printWeights(Weights *weights);
