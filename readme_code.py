import os
import sys
from pathlib import Path
import networkx as nx
from scripts.config import Config
from scripts.convert_to_nx import NxGraph
from scripts.graph_embedding import Node2Vec
from scripts.dag import build_dag


default_config_path = 'scripts/configuration.ini'
default_config_path = Path(default_config_path).resolve()

## Run the following code block to reset the configuration file
class_config = Config()
class_config.reset()
class_config.__init__(config_path = default_config_path) 

## Load configuration.ini into a dictionary
params = Config(config_path = default_config_path).params

if not os.path.exists(params['save_path']):
	os.makedirs(params['save_path'])
	
## Build dag from HPO
sys.path.append(params['main_dir'])
hpo_dag = build_dag()

## Convert dag to an nx object
nx_object = NxGraph(hpo_dag, directed = False, weight_system = params['weight_system']) 

nx_object.create_graph()
hpo_nx_graph = nx_object.graph
hpo2vec = Node2Vec(hpo_nx_graph)

## Train the embedding model (Node2Vec+ if params['extended'] is True)
hpo2vec.run(params['extended'])
hpo2vec.save_embeddings()