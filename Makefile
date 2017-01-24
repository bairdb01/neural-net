compile: node.h network.h weights.h data.h
	gcc -Wall -std=c99 network.c node.c weights.c data.c -o ann -lm

