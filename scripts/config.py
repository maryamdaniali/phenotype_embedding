'''
This module loads and resets configurations stored in configuration.ini, including paths, global variables, etc.
'''
import os
from datetime import datetime
import shutil
from pathlib import Path
import configparser
from configparser import ConfigParser, ExtendedInterpolation
CONFIG_PATH = 'scripts/configuration.ini'


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
            raise ValueError("Failed to open/find config file") from exp

        # Load parameters
        self.params = {}
        for section in config.sections():
            for var_name, var_value in config.items(section):
                if section == 'Node2Vec':
                    if var_name in ['p', 'q', 'learning_rate']:
                        try:
                            var_value = float(var_value)
                        except ValueError:
                            raise ValueError(f"Expected a float for {var_name}, got {var_value}")
                    elif var_name in ['extended']:
                        var_value = var_value.lower()
                        if var_value not in ['true', 'false']:
                            raise ValueError(f"Expected 'true' or 'false' for {var_name}, got {var_value}")
                        var_value = (var_value == 'true')
                    elif var_name in ['weight_system']:
                        valid_weight_systems = ['equal', 'probabilistic', 'probabilistic_with_bias', 'random']
                        if var_value not in valid_weight_systems:
                            raise ValueError(f"Invalid value for {var_name}: {var_value}. Expected one of {valid_weight_systems}")
                    else:
                        try:
                            var_value = int(var_value)
                        except ValueError:
                            raise ValueError(f"Expected an integer for {var_name}, got {var_value}")
                # Validate paths and file names, support Unix and Windows
                elif 'file' in  var_name or  'path' in var_name or 'dir' in var_name:
                    var_value = Path(var_value).resolve()     
                # General boolean checks
                elif var_value.lower() in ['true', 'false']:
                    var_value = (var_value.lower() == 'true')
                # Handle None values
                elif not var_value or var_value == 'None':
                    var_value = None
                # Assign validated value
                self.params[var_name] = var_value
    def reset(self,config_path = CONFIG_PATH):
        """ Reset configuration parameters by first
        creating a backup file and then
        writing the default values to the file and loading them


        Keyword Arguments:
            config_path {str} -- path to the config file
                                 (default: {'scripts/configuration.ini'})
        """

        Config.backup_config(config_path)

        Config.rewrite_config(config_path)
        self.__init__(config_path)

    @staticmethod
    def rewrite_config(config_path):
        """ Write default values to the configuration.ini file

            Keyword Arguments:
            config_path {str} -- path to the config file
        """
        config = configparser.ConfigParser()
        curr_dir = Path('.').absolute()

        if curr_dir.name == 'scripts':
            # adjust path if script is running from the 'scripts' directory
            project_dir = curr_dir.parent
        else:
            project_dir = curr_dir

        config['Node2Vec'] = {
                            '; note1': 'parameters of biased random walks',
                            'p' : '1',
                            'q' :'0.05',
                            'num_walks' : '10',
                            'num_steps' : '5',
                            'vector_length' : '128',
                            'batch_size' : '1024',
                            '; note2': 'parameters of the skip-gram model',
                            'learning_rate' : '0.001',
                            'num_epochs': '50',
                            'num_negative_samples' : '4',
                            '; note3': 'to apply node2vec+ put True',
                            'extended':'False',
                            '; note4': 'weight system use equal, probabilistic, probabilistic_with_bias, or random',
                            'weight_system' : 'equal'
                            }    

        config['Paths'] = { 'main_dir': str(project_dir),
                            'hpo_filename': 'hp-2020-10-12.obo',
                            'hpo_path': "${main_dir}/data/${hpo_filename}",
                            'save_path': "${main_dir}/results/weights_${Node2Vec:weight_system}_extended_${Node2Vec:extended}",
                            'frequecy_filename': 'Table_S2.csv',
                            'frequecy_file': "${main_dir}/data/${frequecy_filename}",
                             }
                            


        config['DAG_Properties'] = {
                                    '; note5': 'to generate the dag for the first time, put False',
                                    'load_saved_dag': 'False',
                                    'dag_filename': "${Paths:save_path}/dag_freq_${Paths:frequecy_filename}_HPO_${Paths:hpo_filename}"
                                    }


        config['Saved_Files'] = {
                                '; note6': ' put True to load previously saved files for the weight_system, hpo_filename, and frequency_file combination, including hpo_graph, hpo_embeddings, dict_hpo, vocabulary_lookup, vocabulary.',
                                'load_saved' : 'False',
                                'hpo_graph_filename': "${Paths:save_path}/hpo_graph",
                                'hpo_embeddings_filename'  :  "${Paths:save_path}/hpo_embeddings",
                                'dict_hpo_filename'  :  "${Paths:save_path}/dict_hpo",
                                'vocabulary_lookup_filename'  :  "${Paths:save_path}/vocabulary_lookup",
                                'vocabulary_filename'  :  "${Paths:save_path}/vocabulary"
                                }

        with open(config_path, 'w') as configfile:
            config.write(configfile)

    @staticmethod
    def backup_config(config_path):
        """ Create a backup from the current config file

        Arguments:
            config_path {string} -- configuration file path
        """
        original_filename = os.path.basename(config_path)
        path_org_file = os.path.dirname(config_path)

        
        backup_path = os.path.join(path_org_file, 'configuration_bk')
        os.makedirs(backup_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_file = os.path.join(backup_path,f"bk_{timestamp}_{original_filename}")

        shutil.copy(config_path, backup_file)
        print(f"Backup configuration.ini created: {backup_file}")
