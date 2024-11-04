""" Main script for creating phenotype embeddings and validating the results
"""
import os
import sys
import random
from tqdm import tqdm
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from config import Config
from convert_to_nx import NxGraph
from graph_embedding import Node2Vec
from dag import build_dag, load_saved_dag

if __name__ == '__main__':

    ## load configuration.ini
    params = Config().params
    if not os.path.exists(params['save_path']):
        os.makedirs(params['save_path'])

    ## build dag from HPO, if not load_saved_dag, build from scratch
    if params['load_saved_dag']:
        hpo_dag = load_saved_dag(params['dag_filename'])
    else:
        hpo_dag = build_dag()


    print(('HP:0002098', 'HP:0001279'), hpo_dag.least_common_ancestor('HP:0002098', 'HP:0001279'))
    print(('HP:0010819', 'HP:0003680') , hpo_dag.least_common_ancestor('HP:0010819', 'HP:0003680') )

    print(hpo_dag.least_common_ancestor_node('HP:0002098', 'HP:0001279') )
    print(hpo_dag.least_common_ancestor_node('HP:0010819', 'HP:0003680') )


    ## convert dag to nx object
    nx_object = NxGraph(hpo_dag, directed = False, weight_system = params['weight_system'])
    nx_object.create_graph()
    hpo_nx_graph = nx_object.graph
    hpo2vec = Node2Vec(hpo_nx_graph)

    ## load trained model or train a new
    if params['load_saved']:
        hpo2vec.load_saved()
    else:
        hpo2vec.run(params['extended'])
        hpo2vec.save_embeddings()

    ## validate similarities
    all_nodes = hpo_dag.descendants('HP:0000001')

    # main sub-graph
    root_id = hpo2vec.get_node_id_by_title('Neonatal seizure')
    all_children = hpo_dag.descendants(root_id)
    num_selected_nodes = len(all_children)
    sim_subtree= hpo2vec.find_smilarity(all_children, all_children)
    subtree_sim_val = list(sim_subtree.values())
    avg_subtree = np.array(subtree_sim_val).mean()
    # randomly selected nodes
    random_nodes = random.sample(all_nodes, num_selected_nodes)
    sim_random_children= hpo2vec.find_smilarity(random_nodes, all_children)
    random_children_sim_val = list(sim_random_children.values())
    avg_random_children = np.array(random_children_sim_val).mean()

    # a non-overlapping sub-graph
    root_id2 = hpo2vec.get_node_id_by_title('Abnormal eye morphology')
    other_children = hpo_dag.descendants(root_id2)
    sim_other= hpo2vec.find_smilarity(all_children, other_children)
    other_sim_val = list(sim_other.values())
    avg_other = np.array(other_sim_val).mean()

    print(f'Average similarity within the sub-graph: {avg_subtree:,.4f}')
    print(f'Average similarity with random nodes: {avg_random_children:,.4f}')
    print(f'Average similarity with a non-overlapping sub-graph: {avg_other:,.4f}')

    ## get simirality of all phenotypes
    similarity_value = []
    similarity_nodes = []
    for curr_node in tqdm(all_nodes,
                        desc = 'Calculating the similarities.', position=0, leave=True):
        similarity_value.append(hpo2vec.find_smilarity_fast([curr_node], all_nodes))
        similarity_nodes.append(curr_node)

    similarity_nodes = np.asarray(similarity_nodes)
    similarity_value = np.asarray(similarity_value)

    is_extended = params['extended']
    save_path = params['save_path']
    outfile = os.path.join(save_path,f"similarity_extended_all_nodes_{is_extended}_weight_{params['weight_system']}.npz")
    np.savez(outfile,similarity_value,similarity_nodes)

    print('Experiments in `main.py` Completed.')
    