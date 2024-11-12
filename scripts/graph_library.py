import heapq
from math import log
import numpy as np
from itertools import combinations, product
from multiprocessing import Pool
from random import choice, randint, seed
import pandas as pd
import copy
from tqdm import tqdm
from scripts.mist import Mist
from scripts.sparse2d import Sparse2D


class Node:
    """
    A class to represent a node (phenotypes) in a Directed Acyclic Graph (DAG), HPO in this case.
    
    Attributes:
        key (str): The unique identifier of the node.
        name (str): A name associated with the node, HPO code.
        df (list): A list representing the discovery and finish time of the node for depth-first search (DFS).
        mist (Mist): A data structure to store a minimum set of disjoint intervals.
        parents (list): List of keys of the parent nodes.
        children (list): List of keys of the child nodes.
        prob (float): Probability associated with the node.
        prob_descendants (float): Cumulative probability considering the descendants of the node.
        neg_log_likelihood (float): Negative log-likelihood based on the node's cumulative probability.
    """
    def __init__(self):

        self.key = ''                
        self.name = ''               
        self.df = [0,0]             
        self.mist = Mist() # Minimum Interval Set Tree for the node
        self.parents = []            
        self.children = []           
        self.prob = 0               
        self.prob_descendants = 0    
        self.neg_log_likelhood = 0  

    def __eq__(self, other):
        """Defines equality based on the node key."""
        return self.key == other.key

    def __lt__(self, other):
        """Defines comparison based on negative log-likelihood."""
        return self.neg_log_likelhood > other.neg_log_likelhood
        
    def __str__(self): 
        """String representation of the Node."""
        return str("Parents: {} > Self: {}, cnt:{}, cnt_desc: {}, neg_log_likelhood: {}, Prob_descendants: {}, DF:{}, MIST: {}, Children: {} ".format(self.parents, self.key, self.cnt, self.cnt_descendants, self.neg_log_likelhood, self.prob_descendants, self.df, self.mist , self.children))


    def get_row(self):
        """Returns a list representing key information about the node."""
        return [self.key, self.prob, self.prob_descendants, self.neg_log_likelhood, self.name]


class PriorityQueue:                                            
    """
    A priority queue implementation using a heap to store nodes, prioritized by negative log-likelihood.
    
    Attributes:
        __pq (list): Internal priority queue storing nodes as a min-heap.
        __set (set): A set to keep track of node keys that are already in the queue.
    """
    def __init__(self):
        self.__pq = []
        self.__set = set()

    def __str__(self):
        """Returns the string representation of the priority queue."""
        return self.printdf()

    def insert(self, node):
        """Inserts a node into the priority queue."""
        if not node.key in self.__set:
            self.__set.add(node.key)
            heapq.heappush(self.__pq , node)

    def pop(self):
        """Removes and returns the node with the highest priority (lowest negative log-likelihood)."""
        val =  heapq.heappop(self.__pq)
        return val

    def printdf(self):
        """Prints the keys of all nodes in the queue along with their negative log-likelihood."""
        prnt_me = '['
        for n in self.pq:
            prnt_me += n.key + ' neg_log_likelhood {' + str(n.neg_log_likelhood) + '} ,'
        return prnt_me + ']'


class Dag:                                                      
    """
    A class to represent a Directed Acyclic Graph (DAG).

    Attributes:
        __nodes (dict): Dictionary storing node objects, keyed by their node key.
        __path (list): List to track the DFS path.
        __d_key (dict): Dictionary to map discovery values to node keys.
        __key_errors (set): Set to store any node key errors.
    """
    def __init__(self):
        self.__nodes = {}
        self.__path  = []
        self.__d_key = {} 
        self.__key_errors = set()
                    
    def __str__(self):
        return str(self.__path)

    def __len__(self): 
        """Returns the number of nodes in the DAG."""
        return len(self.__nodes)

    def __getitem__(self, item):
        """Returns a node by key or discovery index."""
        if type(item) == int:
            return self.__nodes[self.__d_key[item]] 
        else:
            try:
                return self.__nodes[item]      
            except:
                return Node()


    def get_all_nodes(self):
        """Returns a list of all node keys in the DAG."""
        _nodes = []
        for k, node in self.__nodes.items():
            _nodes.append(k)
        return _nodes

    def delete(self, node_key):
        """Deletes a node from the DAG."""
        del self.__nodes[node_key]

    def append(self, node_key):
        """Appends a new node to the DAG."""
        if node_key not in self.__nodes:
            self.__nodes[node_key] = Node()

    def set_node_key(self, node_key):
        """Sets the key for a specific node."""
        self.__nodes[node_key].key = node_key

    def add_node_child(self, node_key, child_key):
        """Adds a child to a specific node."""
        self.__nodes[node_key].children.append(child_key)

    def add_node_parent(self, node_key, parent_key):
        """Adds a parent to a specific node."""
        self.__nodes[node_key].parents.append(parent_key)

    def random_node(self):
        """Returns a random node key from the DAG."""
        return choice(list(self.__nodes))

    def set_name(self, node_key, name):
        """Sets the long name for a specific node."""    
        if node_key in self.__nodes:
            self.__nodes[node_key].name = name

    def get_name(self, node_key):
        """Returns the long name of a specific node."""
        return self.__nodes[node_key].name


    def get_rows(self):
        """Returns a DataFrame containing key attributes of all nodes in the DAG."""
        rows = []
        for k, node in self.__nodes.items():
            rows.append(node.get_row())

        data_frame = pd.DataFrame(rows, 
        columns =[ 'HPO_CODE'
        , 'Base probability'
        , 'Propagated probability' 
        , 'neg_log_likelhood'
        , 'name'
        ] )
        return data_frame

    def set_dkey(self):    
        """Creates a hash table that maps discovery values to node keys."""                                    
        for k, node in self.__nodes.items():
            self.__d_key[ node.df[0] ] = node.key


    def get_dkey(self, hpo_key):   
        """Returns the discovery value of a given node."""                                     
        return self.__nodes[hpo_key].df[0]  
   

    def least_common_ancestor(self, node1_key, node2_keys, pq=None, d=None):
        """
        Finds the least common ancestor for a set of nodes in the DAG.

        Args:
            node1_key (str): The key of the first node.
            node2_keys (list or str): A list or single key of the second node(s).
            pq (PriorityQueue): A priority queue for storing nodes during traversal.
            d (list): List of discovery values for node2(s).

        Returns:
            float: The negative log-likelihood of the LCA node.
        """
        if pq == None:
            pq =  PriorityQueue()
            pq.insert(self.__nodes[node1_key])
            d=[]
            if type(node2_keys) is list:
                for node2_key in node2_keys:
                    d.append(self.__nodes[node2_key].df[0])
            else:
                d.append(self.__nodes[node2_keys].df[0]) 
        val = pq.pop()
        if val.mist.intersects(d):
            return val.neg_log_likelhood
        for parent in val.parents:
            pq.insert(self.__nodes[parent])
        return self.least_common_ancestor(node1_key, node2_keys, pq, d)

    def least_common_ancestor_node(self, node1_key, node2_keys, pq=None, d=None):
        """
        Finds the Least Common Ancestor (LCA) of two nodes in the DAG.
        
        Parameters:
        node1_key (str): The key of the first node.
        node2_keys (str or list): The key(s) of the second node or list of nodes to compare.
        pq (PriorityQueue, optional): Priority queue for traversing the nodes.
        d (list, optional): List of discovery values for the second node(s).

        Returns:
        str: The key of the LCA node.
        """

        if pq == None:
            pq =  PriorityQueue()
            pq.insert(self.__nodes[node1_key])
            d=[]
            if type(node2_keys) is list:
                for node2_key in node2_keys:
                    d.append(self.__nodes[node2_key].df[0])
            else:
                d.append(self.__nodes[node2_keys].df[0]) 
        val = pq.pop()
        if val.mist.intersects(d):
            return val.key
        for parent in val.parents:
            pq.insert(self.__nodes[parent])
        return self.least_common_ancestor_node(node1_key, node2_keys, pq, d)

    def roots(self):
        """
        Finds all the root nodes in the DAG (nodes without parents).

        Returns:
        list: A list of keys corresponding to root nodes.
        """
        roots = []
        for k,v in self.__nodes.items():
            if len(v.parents) == 0:
                roots.append(k)
        return roots

    def depth_first_label(self):
        """
        Performs a Depth First Search (DFS) to label the nodes in the DAG with 
        discovery and finishing times, and updates the path for traversal.
        """
        d = 0
        stack = self.roots()
        self.__path.append(stack[-1])
        while stack:
            self.__nodes[stack[-1]].df[1] = d
            child = self.child1(stack[-1])
            if child:
                d += 1
                stack.append(child)
                self.__nodes[stack[-1]].df[0] = d
                self.__path.append(stack[-1])
            else:
                stack.pop()

    def set_node_attributes(self, frequencies):
        """
        Sets the probabilities and other attributes for each node in the DAG 
        based on given frequencies and initializes them.

        Parameters:
        frequencies (dict): A dictionary containing frequency data for nodes.
        """
        print( "Setting probablities for {} phenotypes inside the DAG".format(len(frequencies)))

        # Initialize the attributes for all nodes in HPO
        for k, node in self.__nodes.items():
            self.__nodes[k].prob = 0
            self.__nodes[k].prob_descendants = 0
            self.__nodes[k].neg_log_likelhood = 0
            self.__nodes[k].df = [0,0]
        print("Finished initializing HPO nodes.")

    
        ## Set the probabilities based on the provided frequencies in DAG
        for k, node in self.__nodes.items():
            node.key = k
            phenotype = frequencies.get(k)
            self.__nodes[k].prob = float(phenotype.get('freq')) if phenotype else 0
            self.__nodes[k].prob_descendants = float(phenotype.get('propagated_freq')) if phenotype  else 0
            if self.__nodes[k].prob_descendants == 0:
                self.__nodes[k].neg_log_likelhood = 0 
            else:
                self.__nodes[k].neg_log_likelhood = - log(self.__nodes[k].prob_descendants, 2)

        self.depth_first_label()
        self.set_dkey()
        for k, node in self.__nodes.items():
            self.__nodes[k].mist.merge(self.bld_mist(k))

        print( "Finished Updating DAG Attributes.")

    
    def bld_mist(self, node):     
        """
        Builds the Modified Interval Set Tree (MIST) for a given node based on its descendants.
        
        Parameters:
        node (str): The key of the node to build the MIST for.

        Returns:
        Mist: The MIST structure for the node.
        """                       
        mist = Mist()
        for descendants in self.descendants(node):
            mist.insert(self.__nodes[descendants].df)
        return mist

    def descendants(self, node_key):   
        """
        Recursively finds all descendants of a given node.

        Parameters:
        node_key (str): The key of the node to find descendants for.

        Returns:
        list: A list of descendant node keys including the node itself.
        """
                    
        descendants = [node_key]
        for child_key in self.__nodes[node_key].children:
            descendants += self.descendants(child_key)
        return list(set(descendants))

    def ancestors(self,node_key):   
        """
        Recursively finds all ancestors of a given node.

        Parameters:
        node_key (str): The key of the node to find ancestors for.

        Returns:
        list: A list of ancestor node keys including the node itself.
        """                      
        ancestors = [node_key]
        for parent_key in self.__nodes[node_key].parents:
            ancestors += self.ancestors(parent_key)
        return list(set(ancestors))

    def child1(self,node):    
        """
        Finds the first child of a node that has not been labeled with discovery time.
        
        Parameters:
        node (str): The key of the node to check for children.

        Returns:
        str: The key of the first unlabeled child, or None if all children are labeled.
        """                            
        for child_key in self.__nodes[node].children:
            child = self.__nodes[child_key]
            if child.df[0] == 0:
                return child_key
        return
    
    def print_roots(self):
        for k,v in self.__nodes.items():
            if len(v.parents) == 0:
                print(self.__nodes[k])

    def print_leaves(self):
        for k,v in self.__nodes.items():
            if len(v.children) == 0:
                print(self.__nodes[k])

    def print_nleaves(self):
        i=0
        for k,v in self.__nodes.items():
            if len(v.children) == 0 and len(v.parents) ==  1:
                i += 1
        print(i)

    def print_nnodes(self):
        i=0
        for k,v in self.__nodes.items():
            i += 1
        print(i)

    def print_nodes(self):
        for k,v in self.__nodes.items():
            print(self.__nodes[k])

    def print_node(self, _key):
        print(self.__nodes[_key])