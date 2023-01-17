""" Takes a graph (dag) of type DAG represented by a class
    created by ADS group and converts it into a nx graph

Returns:
    netwrokx graph -- a graph of type nx
"""
import pickle
import random
import networkx as nx
import config
params = config.Config().params



class NxGraph():
    """A class that converts a DAG represented by a class created by ADS to a nx graph
    """
    def __init__ (self, dag, directed = False, weight_system = 'probabilistic'):
        """ Initialize NxGraph class

        Arguments:
            dag {DAG} -- graph of type DAG in graph_library

        Keyword Arguments:
            directed {bool} -- if the  return graph should be directed (default: {False})
            weight_system {str} -- weight system for edges could be 'equal' to 1
                                or 'random' to assign a random probability
                                or 'probabilistic' based on frequency of nodes in the corpus
                                and 'probabilistic_with_bias' frequency with 1 offset
                                (default: {'probabilistic'})
        """
        self.dag = dag
        self.graph = None
        self.all_nodes = dag.get_all_nodes()
        self.num_nodes = len(self.all_nodes)
        self.directed = directed #could be True or False
        self.weight_system = weight_system #could be 'equal' or 'probabilistic'


    def create_graph(self):
        """Create the nx graph
        """
        self.build_initial_graph()
        self.add_nodes()
        self.get_edges_from_dag()
        print('Dag converted into a NetworkX graph')


    def save_graph(self):
        """Save nx graph in a pickle file
        """
        filename = params['save_path']+'/nx_graph'
        print('Saving nx_graph:',filename)
        with open(filename,'wb') as outfile:
            pickle.dump(self.graph, outfile)

    def build_initial_graph(self):
        """Build an initial nx graph could be directed or not
        """
        graph = nx.DiGraph() if self.directed else nx.Graph()
        self.graph = graph

    def get_nodes_from_dag(self):
        """List nodes and their properties from dag

        Returns:
            list of tuple -- each tuple contains node key and other properties including
                            name,
                            cnt (frequency),
                            prob (of the node),
                            prob_descendants (probability of its descendants)
        """
        nodes_list = []
        for i in range(self.num_nodes):
            curr_node = self.all_nodes[i]
            node = self.dag.__getitem__(curr_node)

            nodes_list.append((node.key,{"name": node.name,
                                        "cnt": node.cnt,
                                        'prob':node.prob,
                                        'prob_descendants': node.prob_descendants}))
        return nodes_list

    def add_nodes(self):
        """Import nodes and their properties to nx graph
        """
        nodes_list = self.get_nodes_from_dag()
        self.graph.add_nodes_from(nodes_list)

    def find_default_weight(self):
        """Find the default edge weight based on difference between
            max and second max node probability value

        Returns:
            float -- min_diff
        """
        prob_des = []
        for i in range(self.num_nodes):
            curr_node = self.all_nodes[i]
            node = self.dag.__getitem__(curr_node)
            prob_des.append(node.prob_descendants)

        set_probs = set(prob_des)
        real_max = max(set_probs)
        set_probs.remove(real_max)
        second_max = max(set_probs)
        min_diff = real_max- second_max
        return min_diff

    def get_edges_from_dag(self):
        """Get edges from dag and import them to nx graph
        """
        default_weight = self.find_default_weight()
        for i in range(self.num_nodes):
            curr_node = self.all_nodes[i]
            node = self.dag.__getitem__(curr_node)
            edges = []
            for child in node.children:
                child_node = self.dag.__getitem__(child)
                if self.weight_system == 'equal':
                    weight = 1
                elif self.weight_system == 'random':
                    weight = random.uniform(0, 1)
                    # weight = abs(random.gauss(0,1))
                elif self.weight_system == 'probabilistic':
                    weight = default_weight + \
                             min(node.prob_descendants, child_node.prob_descendants)
                elif self.weight_system == 'probabilistic_with_bias':
                    weight = 1 + default_weight + \
                             min(node.prob_descendants, child_node.prob_descendants)

                else:
                    print('weight_system not supported, used 1s')
                    weight = 1
                edges.append((node.key,child,{'weight': weight}))
            self.add_edges(edges)

    def add_edges(self, edges):
        """Add a list of edges to the nx graph

        Arguments:
            edges {list nx edge}
        """
        self.graph.add_edges_from(edges)

    def print_graph_info(self):
        """Print nx graph general info on terminal
        """
        print(nx.info(self.graph))
