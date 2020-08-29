from glob import glob
import configparser
import sys
import os

# import code for evaluating radial models
from e3nn.radial import *
from laurent import *
from functools import partial


# parses a delimited string
def parse_list(s, separator=",", func=None):
    fields = s.split(separator)
    return_list = [ i.strip() for i in fields ]
    if func is not None:
        return_list = list(map(func, return_list))
    return return_list


class Config:

    def atomic_number(self, e):
        if isinstance(e, str):
            if e.isnumeric(): return int(e)
            return self.symbol_to_number[e]
        else:
            return e

    # parses config files
    # later filenames overwrite values from earlier filenames
    # (so you can have mini configs that change just a few settings)
    def __init__(self, *filenames):
        config = configparser.ConfigParser()

        if len(filenames) == 0:
            filenames = ["training.ini"]
            if len(sys.argv) > 1:
                filenames += sys.argv[1:]
                
        config.read(filenames)

        # where to do the training
        self.device = config['general']['device']

        self.symbol_to_number = {}
        self.number_to_symbol = {}
        # dictionaries between symbols and numbers
        #print("Building atomic symbol dictionary...")
        for symbol, num_str in config.items('symbols_numbers_dict'):
            symbol = symbol.title()
            num = int(num_str)
            self.symbol_to_number[symbol] = num
            self.number_to_symbol[num] = symbol
        #print(self.symbol_to_number)
        
        # all expected elements
        self.all_elements = [self.atomic_number(e) for e in parse_list(config['general']['all_elements'])]
        assert len(self.all_elements) == len(set(self.all_elements)), "duplicate element"
        self.n_elements = len(self.all_elements)

        # which elements to predict NMR shieldings for
        self.relevant_elements = [self.atomic_number(e) for e in parse_list(config['general']['relevant_elements'])]
        for e in self.relevant_elements:
            assert e in self.all_elements, f"relevant element {e} not found in all_elements"
        assert len(self.relevant_elements) == len(set(self.relevant_elements)), "duplicate element"

        # where the raw data are stored
        self.data_source = config['data']['source']
        if self.data_source.startswith('hdf5'):
            if self.data_source == 'hdf5_0':
                self.file_format = 0
            else:
                self.file_format = 1
            self.data_source = 'hdf5'
            self.hdf5_filenames = list(sorted(glob(config['data']['hdf5_filenames'])))
            assert len(self.hdf5_filenames) > 0, "no files found!"
        elif self.data_source == 'SQL':
            self.connect_params = dict(config['connect_params'])
            self.SQL_fetch_size = int(config['data']['SQL_fetch_size'])



        # how many jiggles to get per file
        # this is not checked--requesting an invalid number will cause a runtime error
        self.jiggles_per_molecule = int(config['data']['jiggles_per_molecule'])

        # number of examples for test-train split
        self.testing_size = int(config['data']['testing_size'])
        self.training_size = int(config['data']['training_size'])
        if 'test_train_shuffle' in config['data']:
            self.test_train_shuffle = config['data']['test_train_shuffle']
        else:
            self.test_train_shuffle = None


        #randomize_training = eval(config['data']['randomize_training'])

        # number of concurrent processes that create DataNeighbors
        self.n_molecule_processors = int(config['data']['n_molecule_processors'])

        # queue capacities:
        self.molecule_queue_cap = int(config['data']['molecule_queue_cap'])
        self.example_queue_cap = int(config['data']['example_queue_cap'])
        self.batch_queue_cap = int(config['data']['batch_queue_cap'])

        # model parameters
        if 'load_model_from_file' not in config['model']:
            self.load_model_from_file = False
        else:
            self.load_model_from_file = config['model']['load_model_from_file']
            if self.load_model_from_file.lower() == "false" or self.load_model_from_file.lower() == "none":
                self.load_model_from_file = False

        self.n_norm = float(config['model']['n_norm'])

        # evaluate model kwargs
        self.model_kwargs = {key:eval(value) for (key,value) in config['model'].items()
                        if key not in {'load_model_from_file'}}
        self.Rs_in = [ (self.n_elements, 0, 1) ]  # n_features, rank 0 tensor, even parity
        self.Rs_out = [ (1,0,1) ]            # one output per atom, rank 0 tensor, even parity
        self.model_kwargs['Rs_in'] = self.Rs_in
        self.model_kwargs['Rs_out'] = self.Rs_out
        self.max_radius = self.model_kwargs['max_radius']


        # training parameters
        self.n_epochs = int(config['training']['n_epochs'])                        # number of epochs
        self.batch_size = int(config['training']['batch_size'])                    # minibatch sizes
        self.testing_interval = int(config['training']['testing_interval'])        # compute testing loss every n minibatches
        self.checkpoint_interval = int(config['training']['checkpoint_interval'])  # save model every n minibatches
        self.checkpoint_prefix = config['training']['checkpoint_prefix']           # save checkpoints to files starting with this
        self.learning_rate = float(config['training']['learning_rate'])            # learning rate


