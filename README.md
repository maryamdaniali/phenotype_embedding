
# Phenotype Embedding using graph representation learning
The goal of this project is to learn phenotype representation with vector embedding by applying a graph representation learning algorithm, i.e., [Node2Vec](https://dl.acm.org/doi/abs/10.1145/2939672.2939754), on the Human Phenotype Ontology (HPO) Directed Acyclic Graph (DAG). Here, we present a novel approach to phenotype representation by incorporating phenotypic frequencies based on 53 million full-text health care notes from more than 1.5 million individuals. We demonstrate the efficacy of our proposed phenotype embedding technique by comparing our work to existing phenotypic similarity-measuring methods. Using phenotype frequencies in our embedding technique, we are able to identify phenotypic similarities that surpass current computational models. Furthermore, our embedding technique exhibits a high degree of agreement with domain experts' judgment. By transforming complex and multidimensional phenotypes from the HPO format into vectors, our proposed method enables efficient representation of these phenotypes for downstream tasks that require deep phenotyping. This is demonstrated in a patient similarity analysis and can further be applied to disease trajectory and risk prediction.

The preprint is available on bioRxiv:
- Daniali, M., Galer, P. D., Lewis-Smith, D., Parthasarathy, S., Kim, E., Salvucci, D. D., ... & Helbig, I. (2022). Enriching Representation Learning Using 53 Million Patient Notes through Human Phenotype Ontology Embedding. [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.07.20.500809.abstract).


Given an input graph <img  src="https://render.githubusercontent.com/render/math?math=G =(V,E)">, where <img src="https://render.githubusercontent.com/render/math?math=V"> is a set of vertices (also known as nodes) and <img src="https://render.githubusercontent.com/render/math?math=E"> is a set of edges, the Node2Vec algorithm finds an embedding space to map the nodes based on their neighborhood connectivity. In doing so, Node2Vec:

<ol>
<li>Generates biased random walks to find the neighborhood of each node</li>
<li>Finds positive and negative samples to learn the negative sampling objective function</li>
<li>Trains the skip-gram (SG) model to find similar nodes given a node</li>
<li>The learned weights can serve as the embedding vectors</li>
</ol>

![model](model.png)

We apply a similar model to phenotype embeddings. In order to get a better representation of phenotypes, this repositoy supports a probabilistic edge weighting method for the input graph. In doing so, it determines the frequency of each node (phenotype) in a given corpus, here a database table of patients and their HPO terms, and calculates the cumulative probability of its descendants. Then, an edge <img  src="https://render.githubusercontent.com/render/math?math=e"> connecting nodes <img  src="https://render.githubusercontent.com/render/math?math=n_1"> and <img  src="https://render.githubusercontent.com/render/math?math=n_2"> has a weight equal to <img  src="https://render.githubusercontent.com/render/math?math=min(p_{n_1}, p_{n_2})">, where <img  src="https://render.githubusercontent.com/render/math?math=p_{i}"> is the cumulative probability of node <img  src="https://render.githubusercontent.com/render/math?math=i">. In this method, by default, all edges have a weight equal to the probability difference between the two most common nodes so that all nodes could be accessible in the biased random walks.

The user can specify the following set of parameters or follow the default values given in `configuration.ini` :
- Paths: list of paths for loading modules, HPO files, and saving directory
- DAG properties: properties of the HPO DAG
- Node2Vec: parameters of the embedding model
- Saved_Files: paths and names of the saved files used for loading the model

In case of accidental and unwanted change in `configuration.ini`, Users can call the`reset()` method in the `config.py` module to reset to the default configurations.

  

## Requirements
Install the packages in requirements/requirements.txt
```bash
pip install -r requirements.txt
```
Or use the conda package list available in `requirements/env.yml`
```bash
conda env create -f env.yml
```

## Usage
In order to get the embedding, the user can follow the code below ( additional functionalities are avaialable in `main.py`):

```python
import os
import sys
import networkx as nx
import config
import convert_to_nx
import graph_embedding

#load configuration.ini
params = config.Config(config_path='./configuration.ini').params

if  not os.path.exists(params['save_path']):
	os.makedirs(params['save_path'])
	
#build dag from HPO
sys.path.append(params['scripts_path'])

import dag
hpo_dag = dag.build_patient_dag()

#convert dag to nx object
nx_object = convert_to_nx.NxGraph(hpo_dag, directed = False, weight_system = 'probabilistic')

nx_object.create_graph()
hpo_nx_graph = nx_object.graph
hpo2vec = graph_embedding.Node2Vec(hpo_nx_graph)

#train the embedding model, Node2Vec+
hpo2vec.run(params['extended'])
hpo2vec.save_embeddings()
```

### Evaluation
A simple evaluation is provided in `main.py` that compares the average cosine similarity of nodes in a subgraph with some randomly selected nodes.
Also, phenotypes in the multidimensional embedding space can be viewed in Tensorboard.
