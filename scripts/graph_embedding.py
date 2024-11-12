""" graph_embedding module contains the Node2Vec class and its method to train
    a skip-gram  model for learning the node embedding
"""
import os
import io
import pickle
import random
from datetime import datetime
from collections import defaultdict
from numba import jit
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scripts.config import Config
params = Config().params


default_params = { 'p':1, 'q':0.05 ,'num_walks':10, 'num_steps':5,
                'vector_length':128,'batch_size':1024,
                'learning_rate':0.001, 'num_epochs':15,
                'num_negative_samples':4 }

class Node2Vec():
    """ Class for representation learning based on Node2Vec algorithm
        Reference:
        Grover, Aditya, and Jure Leskovec.
        "node2vec: Scalable feature learning for networks."
        Proceedings of the 22nd ACM SIGKDD international conference
        on Knowledge discovery and data mining. 2016.
    """
    def __init__(self, graph):
        """Inits Node2Vec class with an nx graph

        Arguments:
            graph {nx graph} -- a graph of type networkX
        """
        self.graph = graph
        self.nodes_dict = None
        self.vocabulary = None
        self.vocabulary_size = None
        self.vocabulary_lookup = None
        self.graph_embeddings = None
        self.avg_weights = self.generate_average_weights()

        # parameters of biased random walk
        self.p = None               # random walk return parameter,
                                    # controls the likelihood of immediately revisiting a node,
                                    # higher value => exploration,
                                    # lower value => keep the walk local

        self.q = None               # random walk in-out parameter.
                                    # controls the search between inward and outward nodes,
                                    # higher value => visit local nodes (BFS),
                                    # lower value => visit further nodes (DFS)

        self.num_walks = None       # number of iterations of random walks.
        self.num_steps = None       # number of steps of each random walk.
        self.vector_length = None   # vector length equal to embedding dimention
        self.batch_size = None      # batch size for creating the dataset

        # parameters of the skip-gram model
        self.learning_rate = None
        self.num_epochs = None
        self.num_negative_samples = None

        self.model = None
        self.model_name = 'my_model_acc'
        self.log_dir = create_log_dir()

    def set_parameters_from_config(self):
        """Set the parameters reading values config file
        """
        self.p = params['p']
        self.q = params['q']
        self.num_walks = params['num_walks']

        self.num_steps = params['num_steps']
        self.vector_length = params['vector_length']
        self.batch_size = params['batch_size']

        # parameters of the skip-gram model
        self.learning_rate = params['learning_rate']
        self.num_epochs = params['num_epochs']
        self.num_negative_samples = params['num_negative_samples']

    def set_parameters(self, param_dict):
        """Set the parameters reading from a dictionary

        Arguments:
            param_dict {dictionary} -- a dictionary contianing parameters and their values
        """
        self.p = param_dict['p']
        self.q = param_dict['q']
        self.num_walks = param_dict['num_walks']

        self.num_steps = param_dict['num_steps']
        self.vector_length = param_dict['vector_length']
        self.batch_size = param_dict['batch_size']

        # parameters of the skip-gram model
        self.learning_rate = param_dict['learning_rate']
        self.num_epochs = param_dict['num_epochs']
        self.num_negative_samples = param_dict['num_negative_samples']

    def build_vocabulary(self):
        """ build the vocabulary from nodes in the graph plus one word for unseen words ("NA")
        """
        self.vocabulary = ["NA"] + list(self.graph.nodes)
        self.vocabulary_lookup = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.vocabulary_size= len(self.vocabulary)

    def get_average_weight(self, node):
        """ return the average weight of surrounding nodes
        for a given node
        """
        neighbors = list(self.graph.neighbors(node))
        weights = []
        for neighbor in neighbors:
            weights.append(self.graph[node][neighbor]["weight"])
        return sum(weights)/len(weights)

    def generate_average_weights(self):
        """ get the average weight of surrounding nodes
        for each node in a dictionary format
        """
        nodes = list(self.graph.nodes())
        dict_avg_weights = {}
        for node in nodes:
            dict_avg_weights[node] = self.get_average_weight(node)
        return dict_avg_weights


    def next_step(self, previous, current):
        """ Probabilistically, select the next visiting node from the current node.
            First adjust the weights of the edges to the neighbors with respect to p and q
            Then, calculate the probability of visiting each neighbor node
            Finally, select one neighbor node

        Arguments:
            previous {nx node} -- previous visited node in the walk
            current {nx node} -- current visiting node in the walk

        Returns:
            [nx node] -- probabilistically selected neighbor node to visit next
        """
        neighbors = list(self.graph.neighbors(current))
        weights = []

        for neighbor in neighbors:
            if neighbor == previous:
                weights.append(self.graph[current][neighbor]["weight"] / self.p)
            elif self.graph.has_edge(neighbor, previous):
                weights.append(self.graph[current][neighbor]["weight"])
            else:
                weights.append(self.graph[current][neighbor]["weight"] / self.q)

        weight_sum = sum(weights)
        probabilities = [weight / weight_sum for weight in weights]
        next_node = np.random.choice(neighbors, size=1, p=probabilities)[0]
        return next_node

    def next_step_extended(self, previous, current):
        """ Extension on next_step based on https://arxiv.org/pdf/2109.08031.pdf
            Accurately Modeling Biased Random Walks on Weighted Graphs Using Node2vec+

        Arguments:
            previous {nx node} -- previous visited node in the walk
            current {nx node} -- current visiting node in the walk

        Returns:
            [nx node] -- probabilistically selected neighbor node to visit next
        """
        neighbors = list(self.graph.neighbors(current))
        weights = []

        for neighbor in neighbors:
            #return case
            if neighbor == previous:
                weights.append(self.graph[current][neighbor]["weight"] / self.p)
            #loosely connected: either connect with a small weight or not connected
            elif previous and \
                (not self.graph.has_edge(previous, neighbor) or \
                self.graph[previous][neighbor]["weight"] < self.avg_weights[neighbor]):
                #noisy case, both loose
                if self.graph[current][neighbor]["weight"] < self.avg_weights[current]:
                    weights.append(self.graph[current][neighbor]["weight"] * min(1,1/self.q))
                # (v,x) tight, (x,t) loose
                else:
                    w_pre_nei = 0 if not self.graph.has_edge(previous, neighbor) \
                                else self.graph[previous][neighbor]["weight"]
                    alpha = 1/self.q + (1- 1/self.q)* w_pre_nei/self.avg_weights[neighbor]
                    weights.append(self.graph[current][neighbor]["weight"] * alpha)
            # strongly connected
            else:
                weights.append(self.graph[current][neighbor]["weight"])
        weight_sum = sum(weights)
        probabilities = [weight / weight_sum for weight in weights]
        next_node = np.random.choice(neighbors, size=1, p=probabilities)[0]
        return next_node

    def random_walk(self, extended = False):
        """ (Biased) Random walk starts from all nodes in graph one by one
            and explores their neighborhood using the next_step function
            in a probabilistic fashion

        Returns:
            list of list -- walks generated from each node in the graph
        """
        walks = []
        nodes = list(self.graph.nodes())

        for walk_iteration in range(self.num_walks):
            random.shuffle(nodes)

            for node in tqdm(
                nodes,
                desc=f"Random walks iteration {walk_iteration + 1} of {self.num_walks}"
                ):
                walk = [node]
                while len(walk) < self.num_steps:
                    current = walk[-1]
                    previous = walk[-2] if len(walk) > 1 else None
                    if extended:
                        next_node = self.next_step_extended(previous, current)
                    else:
                        next_node = self.next_step(previous, current)

                    walk.append(next_node)
                # Replace node ids in the walk with token/lookup ids.
                walk = [self.vocabulary_lookup[token] for token in walk]
                walks.append(walk)

        return walks

    def generate_examples(self, sequences):

        """ Generate negative and positive example to train skip-gram
            using the negative sampling technique
            target:     a node in a walk
            context:    another node in a walk
            label:      1 if target and context are samples from the walks,
                        else 0, for example if were randomly selected
            weight:     count co-occurrence of target and context in walks

        Arguments:
            sequences {list of list} -- walk sequences

        Returns:
            np.array -- targets, array of targets
            np.array -- contexts, array of contexts
            np.array -- labels, array of labels
            np.array -- weights, array of weights
        """
        example_weights = defaultdict(int)
        for sequence in tqdm(sequences, desc = "Generating postive and negative examples"):
            pairs, labels = keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=self.vocabulary_size,
                window_size=self.num_steps,
                negative_samples=self.num_negative_samples
                )
            for idx, pair in enumerate(pairs):
                label = labels[idx]
                target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
                if target == context:
                    continue
                entry = (target, context, label)
                example_weights[entry] += 1
        targets, contexts, labels, weights = [], [], [], []
        for entry in example_weights:
            weight = example_weights[entry]
            target, context, label = entry
            targets.append(target)
            contexts.append(context)
            labels.append(label)
            weights.append(weight)

        return np.array(targets), np.array(contexts), np.array(labels), np.array(weights)

    def create_model(self):
        """ Create the skip-gram model

            Skip-gram is a binary classification model containing:

            1. An embedding look up for the target node
            2. An embedding look up for the context node
            3. Dot product computation between the two embeddings
            4. Apply sigmoind function to the dot product and compare it to the label
            5. Apply binary cross-entropy loss to update the weights

        Returns:
            [tf.keras.model] -- model
        """
        inputs = {
            "target": layers.Input(name="target", shape=(), dtype="int32"),
            "context": layers.Input(name="context", shape=(), dtype="int32"),
        }
        embed_item = layers.Embedding(
                                    input_dim=self.vocabulary_size,
                                    output_dim=self.vector_length,
                                    embeddings_initializer="he_normal",
                                    embeddings_regularizer=keras.regularizers.l2(1e-6),
                                    name="item_embeddings",
                                    )

        target_embeddings = embed_item(inputs["target"])
        context_embeddings = embed_item(inputs["context"])
        logits = layers.Dot(axes=1,
                            normalize=False,
                            name="dot_similarity")([target_embeddings, context_embeddings]
                            )
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def build_nodes_dictionary(self):
        """ Build a mapping dictionary beween nodes names and their ids
        """
        nodes_dict = {}
        for node_id,node_name in nx.get_node_attributes(self.graph,'name').items():
            nodes_dict[node_name]= node_id
        self.nodes_dict = nodes_dict


    def get_node_id_by_title(self,node_title):
        """ A helper function to get node id by its title

        Arguments:
            title {str} -- node title (here, phenotype name)

        Returns:
            nx node id -- node id available in nx graph
        """
        return self.nodes_dict[node_title]

    def get_node_title_by_id(self,node_id):
        """ A helper function to get node title by its id

        Arguments:
            id {nx node id} -- node id available in nx graph

        Returns:
            str -- node title (here, phenotype name)
        """
        return nx.get_node_attributes(self.graph,'name')[node_id]


    def run(self, extended = False, save_file = True):
        """ Main function to initiate, create, and train skip-gram model for node embedding

        Keyword Arguments:
            extended {bool} -- whether to apply extended Node2Vec+ or the basic (default: {False})
            save_file {bool} -- whether to save the trained model (default: {True})
        """
        self.set_parameters_from_config()
        self.build_vocabulary()
        self.build_nodes_dictionary()

        walks = self.random_walk(extended)
        print("\nNumber of walks generated:", len(walks))
        targets, contexts, labels, weights = self.generate_examples(sequences=walks)
        dataset = create_dataset(
            targets=targets,
            contexts=contexts,
            labels=labels,
            weights=weights,
            batch_size=self.batch_size
            )

        model = self.create_model()
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            #metrics=['acc']
            )
        print(model.summary())

        # keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
        tensorboard_callback = \
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        history = model.fit \
            (dataset, epochs=self.num_epochs,callbacks=[tensorboard_callback])

        self.model = model
        Node2Vec.visualize_loss(history)


        self.graph_embeddings = model.get_layer("item_embeddings").get_weights()[0]
        print('Graph embeddings created.')

        if save_file:
            self.save_data()

    @staticmethod
    def visualize_loss(history):
        """Visualize the loss trend in model training

        Arguments:
            history {keras History} -- history of the loss changes
        """
        plt.plot(history.history["loss"])
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.savefig(str(params['save_path']/'model_loss.png'))

    def save_data(self):
        """ A helper function to save files required to run load the saved model
        """
        self.model.save(os.path.join(params['save_path'], self.model_name))
        dict_file_name = {'hpo_graph':self.graph,
                          'hpo_embeddings': self.graph_embeddings,
                          'dict_hpo': self.nodes_dict,
                          'vocabulary_lookup': self.vocabulary_lookup,
                          'vocabulary': self.vocabulary
                         }

        for filename, file in dict_file_name.items():
            with open(os.path.join(params['save_path'], filename), 'wb') as outfile:
                pickle.dump(file,outfile)
            outfile.close()
        print('Data files saved in ', params['save_path'])


        with open(os.path.join(self.log_dir, 'used_config.txt'), 'w') as file_cofig:
            print(params, file=file_cofig)
        print('Config file saved in ', self.log_dir)


    def load_saved(self):
        """ A helper function to load saved files required to run the model
        """
        self.graph = load_pickle(params['hpo_graph_filename'])
        self.graph_embeddings = load_pickle(params['hpo_embeddings_filename'])
        self.nodes_dict = load_pickle(params['dict_hpo_filename'])
        self.vocabulary_lookup = load_pickle(params['vocabulary_lookup_filename'])
        self.vocabulary= load_pickle(params['vocabulary_filename'])
        self.model = tf.keras.models.load_model(os.path.join(params['save_path'], self.model_name))
        print("Loading saved files done.")


    def save_embeddings(self):
        """ Save embedding vectors, embeddings,
            and their corresponding titles, metadata, to .tsv files
            to load in tensorboard for reduced-dimensions visualization
        """

        with io.open(os.path.join(self.log_dir,'embeddings.tsv'), 'w') as out_v,\
             io.open(os.path.join(self.log_dir,'metadata.tsv'), 'w') as out_m:
            for idx, node_id in enumerate(
                                        tqdm(self.vocabulary[1:],
                                        'Saving embedings and their label in .tsv format')):
                node_title = self.get_node_title_by_id(node_id)
                vector = self.graph_embeddings[idx]
                out_v.write("\t".join([str(x) for x in vector]) + "\n")
                out_m.write(node_title + "\n")

        hpo_weights= tf.Variable(self.model.get_layer('item_embeddings').get_weights()[0][1:])
        checkpoint = tf.train.Checkpoint(embedding=hpo_weights)
        checkpoint.save(os.path.join(self.log_dir, 'embedding.ckpt'))
        print(f'Checkpoints are saved in {self.log_dir}')


    def get_batch_embeddings(self, list_node_ids):
        """Find the embedding vectors of on a batch of node ids

        Arguments:
            list_node_ids { list(str)} -- list of nodes id

        Returns:
            [list of list(float)] -- [description]
        """
        query_embeddings = []
        for node_id in list_node_ids:
            token_id = self.vocabulary_lookup[node_id]
            hpo_embedding = self.graph_embeddings[token_id]
            query_embeddings.append(hpo_embedding)

        query_embeddings = np.array(query_embeddings)
        return query_embeddings

    def find_smilarity(self,batch_ids_1, batch_ids_2):
        """ Finds similarity between two batches of nodes id

        Arguments:
            batch_ids_1 {list(str)} -- list of node id
            batch_ids_2 {list(str)} -- list of node id

        Returns:
            list(float) -- similarity value pair-wise between nodes in each batch
        """
        embeddings_1 = self.get_batch_embeddings(batch_ids_1)
        embeddings_2 = self.get_batch_embeddings(batch_ids_2)
        dict_similarities = {}
        for idx_i, e_i in enumerate(embeddings_1):
            for idx_j, e_j in enumerate(embeddings_2):
                similarities = tf.linalg.matmul(
                    tf.math.l2_normalize([e_i]),
                    tf.math.l2_normalize([e_j]),
                    transpose_b = True)
                if batch_ids_1[idx_i]!= batch_ids_2[idx_j]:
                    dict_similarities[(batch_ids_1[idx_i], batch_ids_2[idx_j])] =\
                         similarities.numpy().tolist()[0][0]
        return dict_similarities

    @staticmethod
    @jit(nopython=True)
    def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
        """Find cosine similarity between two numpy arrays
           This  method uses numba to do the calculation in machine code

        Arguments:
            u {np.ndarray} -- source vector
            v {np.ndarray} -- destination vector

        Returns:
            [float] -- cosine similarity between two vectors
        """
        assert u.shape[0] == v.shape[0]
        uv = 0
        uu = 0
        vv = 0
        for i in range(u.shape[0]):
            uv += u[i]*v[i]
            uu += u[i]*u[i]
            vv += v[i]*v[i]
        cos_theta = 1
        if uu!=0 and vv!=0:
            cos_theta = uv/np.sqrt(uu*vv)
        return cos_theta


    def find_smilarity_fast(self,batch_ids_1, batch_ids_2):
        """ Finds similarity between two batches of nodes id
            using cosine_similarity_numba
        Arguments:
            batch_ids_1 {list(str)} -- list of node id
            batch_ids_2 {list(str)} -- list of node id

        Returns:
            list(float) -- similarity value pair-wise between nodes in each batch
        """
        embeddings_1 = self.get_batch_embeddings(batch_ids_1)
        embeddings_2 = self.get_batch_embeddings(batch_ids_2)
        similarity_list = []
        for _, e_i in enumerate(embeddings_1):
            for __, e_j in enumerate(embeddings_2):
                # if batch_ids_1[idx_i]!= batch_ids_2[idx_j]: # removed for speed
                similarity_list.append(Node2Vec.cosine_similarity_numba(e_i,e_j))
        similarity_list = np.asarray(similarity_list)
        return similarity_list

def load_pickle(file_name):
    """Load a pickled file

    Arguments:
        file_name {string} -- filename address

    Raises:
        ValueError: if filename does not exist or could not be opened

    Returns:
        file -- returns loaded pickels file
    """
    try:
        with open(file_name,'rb') as file:
            return pickle.load(file)
    except OSError as exp:
        raise ValueError(f"Failed to open/find {file_name}") from exp

def create_dataset(targets, contexts, labels, weights, batch_size):
    """ Convert the data into tf.data.Dataset objects

    Arguments:
        targets {np.array} -- array of targets (a node in a walk)
        contexts {np.array} -- array of contexts (another node in a walk)
        labels {np.array} -- array of labels
                            (1 if target and context are samples from the walks, else 0)
        weights {np.array} -- array of weights
                            (count co-occurrence of target and context in walks)
        batch_size {int} -- batch size for creating the dataset

    Returns:
        [tf.data.Dataset] -- dataset
    """
    inputs = {
        "target": targets,
        "context": contexts,
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, weights))
    dataset = dataset.shuffle(buffer_size=batch_size * 2)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) #added experimental
    return dataset

def create_log_dir():
    """ Create a log directroy based on date and time
    Returns:
        str -- log directory
    """
    log_dir = os.path.join \
        (params['save_path'], 'log_dir', datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
