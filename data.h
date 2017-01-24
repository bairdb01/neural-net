/*
 * Filename: data.h
 * Author: Ben Baird
 * Date: Jan 22, 2017
 * Description: Header file for data.c
 */

#ifndef STDLIB_H
#define STDLIB_H
#include <stdlib.h>
#endif

#ifndef STDIO_H
#define STDIO_H
#include <stdio.h>
#endif

typedef struct Data {
    int id;
    double *inputs;
    double *targets;
    struct Data *next;
}Data;

typedef struct DataSet {
    int nIn;
    int nTargets;
    int nData;
    Data *data;
}DataSet;

Data * initData(int nInputs, int nTargets, int id);

DataSet * initDataSet(int nInputs, int nTargets);

void printDataSet(DataSet *set);

void freeDataChain(Data *data);

void freeDataSet(DataSet *set);
