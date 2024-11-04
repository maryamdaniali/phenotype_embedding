""" Creates a directed acyclic graph (DAG) from HPO.obo file
    using on modules in the patient-similarity-argo repository
    by ADS and gets nodes frequency based on data given in a csv file.
    See supplementary materials, Table_S2.cvs.
"""
import pickle
from scripts.load_hpo import create_dag
from scripts.load_freq_data import load_freq_data
from scripts.config import Config
params = Config().params



def build_dag(save_file = True, save_path = params['save_path'], hpo_file_path = params['hpo_path']):

    """Create dag based on the hpo.obo file

    Keyword Arguments:
        save_file {bool} -- if True save the file in save_path (default: {True})
        save_path {string} -- path to save directory (default: {params['save_path']})
        hpo_file_path {string} -- path to hpo.obo file (default: {params['hpo_path']})

    Returns:
         dag {DAG} -- returns HPO dag
    """

    # Steps
    # 1. create dag file - pass hpo file
    # 2. update the edge info based on read csv file 
    # 2.1. check what are the node attributes
    # 2.2. check the node attributes used in convert_to_nx


    dag = create_dag(hpo_file_path)

    print('Update dag based on frequency data...')
    freq_dict = load_freq_data(params['frequecy_file'])
    dag.set_node_attributes(freq_dict)


    if save_file:
        dag_filename = params['dag_filename']
        print('Saving dag:',dag_filename)
        with open(dag_filename,'wb') as outfile:
            pickle.dump(dag, outfile)

    return dag

def load_saved_dag(dag_file):
    """Load pre-saved/ pickled dag file

    Arguments:
        dag_file {string} -- path to dag pickle file, available in config

    Returns:
        dag {DAG} -- returns HPO dag
    """
    dag_name = dag_file.name

    try:
        with open(dag_file,'rb') as file:
            print(f'Loading saved Dag: {dag_name}')
            return pickle.load(file)
    except IOError as exp:
        raise ValueError(f"Failed to open/find dag file: {dag_name} in {dag_file.parent}") from exp
