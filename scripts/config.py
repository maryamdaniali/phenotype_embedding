'''
This module loads and resets configurations stored in configuration.ini, including paths, global variables, etc.
'''

from datetime import datetime
import shutil
from pathlib import Path
import configparser
from configparser import ConfigParser, ExtendedInterpolation
CONFIG_PATH = './configuration.ini'


class Config():
    '''
    A class to load and reset configurations stored in .ini format
    '''

    def __init__(self, config_path = CONFIG_PATH):
        """ Loads configuration from a saved file in configparser format

        Keyword Arguments:
        config_path {str} -- path to a configuration file (default: {CONFIG_PATH})

        """
        config = ConfigParser(interpolation=ExtendedInterpolation())
        try:
            with open(config_path) as file:
                config.read_file(file)
        except IOError as exp:
            raise ValueError("Failed to open/find all config files") from exp

        #load paths
        self.params = {}
        for section in config.sections():
            for var_name, var_value in config.items(section):
                if section == 'Node2Vec':
                    if var_name in ['p', 'q', 'learning_rate']:
                        var_value = float(var_value)

                    elif var_name in ['extended']:
                        var_value = (var_value.lower() == 'true')

                    else:
                        var_value = int(var_value)
                elif var_value.lower() == 'true':
                    var_value = True
                elif var_value.lower() == 'false':
                    var_value = False

                elif not var_value or var_value=='None':
                    var_value = None
                self.params[var_name] = var_value
    def reset(self,config_path = './configuration.ini'):
        """ Reset configuration parameters by first
        creating a backup file and then
        writing the default values to the file and loading them


        Keyword Arguments:
            config_path {str} -- path to the config file
                                 (default: {'./configuration.ini'})
        """

        Config.backup_config(config_path)

        Config.rewrite_config(config_path)
        self.__init__(config_path)

    @staticmethod
    def rewrite_config(config_path):
        """ Write default values to the configuration file

            Keyword Arguments:
            config_path {str} -- path to the config file
        """
        config = configparser.ConfigParser()
        curr_dir = Path('.').absolute()

        config['Paths'] = { 'curr_dir': str(curr_dir),
                            'patient_similarity_repo_path': str(curr_dir.parent),
                            'scripts_path': str(curr_dir.parent/'patient-similarity-argo'/'scripts'),
                            'hpo_filename': 'hpo_06_08_2020.obo',
                            'hpo_file': str(curr_dir.parent/'patient-similarity-argo'/'data'/'hpo_06_08_2020.obo'),
                            'save_path': str(curr_dir/'saved_data') }

        config['DAG_Properties'] = {'prune': 'False',
                                    'patients_table_name': 'lab.subset',
                                    'filter_notes': 'False',
                                    'load_saved_dag': 'False',
                                    'dag_filename': str(curr_dir/'saved_data'/'dag_table_lab.subset_pruned_False_hpo_06_08_2020.obo')
                                    }

        config['Node2Vec'] = {'p' : '1',
                            'q' :'0.05',
                            'num_walks' : '10',
                            'num_steps' : '5',
                            'vector_length' : '128',
                            'batch_size' : '1024',
                            'learning_rate' : '0.001',
                            'num_epochs': '15',
                            'num_negative_samples' : '4',
                            'extended':'False'

                            }
        config['Saved_Files'] = {'load_saved_files' : 'True',
                                'hpo_graph_filename': str(curr_dir/'saved_data'/'hpo_graph'),
                                'hpo_embeddings_filename'  :  str(curr_dir/'saved_data'/'hpo_embeddings'),
                                'dict_hpo_filename'  :  str(curr_dir/'saved_data'/'dict_hpo'),
                                'vocabulary_lookup_filename'  :  str(curr_dir/'saved_data'/'vocabulary_lookup'),
                                'vocabulary_filename'  :  str(curr_dir/'saved_data'/'vocabulary')
                                }

        with open(config_path, 'w') as configfile:
            config.write(configfile)

    @staticmethod
    def backup_config(config_path):
        """ Create a backup from the current config file

        Arguments:
            config_path {string} -- configuration file path
        """
        original_filename = config_path.split('/')[-1]
        path_components = config_path.split('/')[:-1]
        path_org_file = '.' if not path_components else ''.join(path_components)
        backup_filename = path_org_file + '/backup_'+\
                        datetime.now().strftime("%Y-%m-%d_%H%M%S") +\
                        '_'+ original_filename
        shutil.copy(original_filename,backup_filename)

