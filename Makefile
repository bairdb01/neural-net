compile: src/node.h src/network.h src/weights.h src/data.h
	gcc -Wall -std=c99 src/network.c src/node.c src/weights.c src/data.c -o ann -lm -g

