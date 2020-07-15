from glob import glob

### configuration ###

# all expected elements
all_elements = ['C', 'H', 'N', 'O', 'S']
n_elements = len(all_elements)

# so we can normalize training data for the nuclei to be predicted
relevant_elements = ['C', 'H']

# where the raw data are stored
hdf5_filenames = list(sorted(glob("data/qm7_*.hdf5")))
assert len(hdf5_filenames) > 0, "no files found!"

# number of examples for test-train split
# the examples picked are strictly in the order they appear
# in the hdf5, but iteration over the set might vary due
# to concurrent processing
testing_size = 1000
training_size = 10000

# number of concurrent processes that create DataNeighbors
n_molecule_processors = 8

# maximum number of pending molecules
molecule_queue_max_size = 1000

# maximum number of pending DataNeighbors
data_neighbors_queue_max_size = 1000

# model parameters
Rs_in = [ (n_elements, 0, 1) ]  # n_features, rank 0 tensor, even parity
Rs_out = [ (1,0,1) ]            # one output per atom, rank 0 tensor, even parity

# training parameters
n_epochs = 2               # number of epochs
batch_size = 100            # minibatch sizes
checkpoint_interval = 10   # save model every checkpoint_interval minibatches
learning_rate = 3e-3       # learning rate
max_radius = 5.0           # consider neighbors out to this radius
n_norm = 14.0              # average number of convolution neighbors per atom

