/*
 * Filename: node.h
 * Author: Ben Baird
 * Date Jan 18, 2017
 * Description: Header file for nodes.c. Contains relevant functions
 *              data structures.
 */
typedef struct Node {
    double weight;
    NodeList * inNodes; // List of nodes to recieve inputs from
    NodeList * outNodes; // List of nodes to output to
} Node;

typedef struct NodeList {
    Node *node;
    struct NodeList *next;
} NodeList;

// Initialize node
Node * initNode();

// Initialize the nodes in reverse order
Node * initNodeList (int numNodes);

// Print a node
void printNode(Node *node);

// Print NodeList
void printNodeList(NodeList *list);

// Free Node
void freeNode(Node* node);

// Free NodeList
void freeNodeList(NodeList *list);
