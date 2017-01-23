/*
 * Filename: data.c
 * Author: Ben Baird
 * Date: Jan 22, 2017
 * Description: Data structure to hold testing/training
 *              data in memory
 */

#ifndef DATA_H
#define DATA_H
#include "data.h"
#endif

Data *initData(int nInputs, int nTargets, int id){
    Data *newData = malloc(sizeof(Data));
    newData->id = id;
    newData->inputs = malloc(sizeof(double)*nInputs);
    newData->targets = malloc(sizeof(double)*nTargets);
    newData->next = NULL;
    return newData;
}

DataSet *initDataSet(int nInputs, int nTargets) {
    DataSet *set = malloc(sizeof(DataSet));
    set->nIn = nInputs;
    set->nTargets = nTargets;
    set->nData = 0;
    set->data = NULL;
    return set;
}

void printDataSet(DataSet *set){
    printf("---- DataSet ----\n");
    printf("In:%d Targets:%d\n", set->nIn, set->nTargets);
    printf("Total: %d\n", set->nData);
    printf("\n");
    Data *temp = set->data;
    while (temp != NULL) {
        printf("Inputs:\n");
        for (int i = 0; i < set->nIn; i++) {
            printf("%lf ", temp->inputs[i]);
        }
        printf("\nTargets\n");
        for (int i = 0; i < set->nTargets; i++) {
            printf("%lf ",temp->targets[i]);
        }
        printf("\n");
        temp = temp->next;
    }
}