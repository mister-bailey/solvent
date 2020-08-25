from glob import glob
import configparser
import sys
import os

config = configparser.ConfigParser()
config_filename = "training.ini"
if len(sys.argv) > 1:
    config_filename = sys.argv[1]
if not os.path.exists(config_filename):
    print(f"Error: configuration file '{config_filename}' not found!")
    exit(1)
config.read(config_filename)

# parses a delimited string
def parse_list(s, separator=",", func=None):
    fields = s.split(separator)
    return_list = [ i.strip() for i in fields ]
    if func is not None:
        return_list = list(map(func, return_list))
    return return_list

# where to do the training
device = config['general']['device']

symbol_to_number = {}
number_to_symbol = {}
# dictionary between symbols and numbers
#print("Building atomic symbol dictionary...")
for symbol, num_str in config.items('symbols_numbers_dict'):
    num = int(num_str)
    symbol = symbol.upper()
#    print(f"{symbol} = {num}")
    symbol_to_number[symbol] = num
    number_to_symbol[num] = symbol

def atomic_number(e):
    if isinstance(e, str):
        if e.isnumeric(): return int(e)
        return symbol_to_number[e]
    else:
        return e

# all expected elements
all_elements = [atomic_number(e) for e in parse_list(config['general']['all_elements'])]
assert len(all_elements) == len(set(all_elements)), "duplicate element"
n_elements = len(all_elements)

# which elements to predict NMR shieldings for
relevant_elements = [atomic_number(e) for e in parse_list(config['general']['relevant_elements'])]
for e in relevant_elements:
    assert e in all_elements, f"relevant element {e} not found in all_elements"
assert len(relevant_elements) == len(set(relevant_elements)), "duplicate element"

# where the raw data are stored
data_source = config['data']['source']
if data_source.startswith('hdf5'):
    if data_source == 'hdf5_0':
        file_format = 0
    else:
        file_format = 1
    data_source = 'hdf5'
    hdf5_filenames = list(sorted(glob(config['data']['hdf5_filenames'])))
    assert len(hdf5_filenames) > 0, "no files found!"
elif data_source == 'SQL':
    connect_params = config['connect_params']
    SQL_fetch_size = int(config['data']['SQL_fetch_size'])



# how many jiggles to get per file
# this is not checked--requesting an invalid number will cause a runtime error
jiggles_per_molecule = int(config['data']['jiggles_per_molecule'])

# number of examples for test-train split
# the examples picked are strictly in the order they appear
# in the hdf5, but iteration over the set might vary due
# to concurrent processing
testing_size = int(config['data']['testing_size'])
training_size = int(config['data']['training_size'])

# number of concurrent processes that create DataNeighbors
n_molecule_processors = int(config['data']['n_molecule_processors'])

# maximum number of pending molecules
molecule_queue_max_size = int(config['data']['molecule_queue_max_size'])

# maximum number of pending DataNeighbors
data_neighbors_queue_max_size = int(config['data']['data_neighbors_queue_max_size'])

# model parameters
load_model_from_file = config['model']['load_model_from_file']
if load_model_from_file.lower() == "false" or load_model_from_file.lower() == "none":
    load_model_from_file = False
Rs_in = [ (n_elements, 0, 1) ]  # n_features, rank 0 tensor, even parity
Rs_out = [ (1,0,1) ]            # one output per atom, rank 0 tensor, even parity
muls = parse_list(config['model']['muls'], func=int)
lmaxes = parse_list(config['model']['lmaxes'], func=int)
max_radius = float(config['model']['max_radius'])
n_norm = float(config['model']['n_norm'])
number_of_basis = int(config['model']['number_of_basis'])
radial_h = int(config['model']['radial_h'])

# training parameters
n_epochs = int(config['training']['n_epochs'])                        # number of epochs
batch_size = int(config['training']['batch_size'])                    # minibatch sizes
testing_interval = int(config['training']['testing_interval'])        # compute testing loss every n minibatches
checkpoint_interval = int(config['training']['checkpoint_interval'])  # save model every n minibatches
checkpoint_prefix = config['training']['checkpoint_prefix']           # save checkpoints to files starting with this
learning_rate = float(config['training']['learning_rate'])            # learning rate

