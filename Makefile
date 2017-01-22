compile: node.h
	gcc -Wall -std=c99 node.c weights.c -o ann -lm
