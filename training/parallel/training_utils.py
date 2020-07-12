from enum import Enum
import numpy as np
import h5py
import torch
torch.set_default_dtype(torch.float64)
import torch_geometric as tg
import e3nn
import e3nn.point.data_helpers as dh

# all expected elements
all_elements = ['C', 'H', 'N', 'O', 'S']
n_elements = len(all_elements)

# so we can normalize training data for the nuclei to be predicted
relevant_elements = ['C', 'H']

# generates one-hots for a list of atomic_symbols
def get_one_hots(atomic_symbols):
    one_hots = []
    for symbol in atomic_symbols:
        inner_list = [ 1. if symbol == i else 0. for i in all_elements ]
        one_hots.append(inner_list)
    return np.array(one_hots)

# compute weights for loss function
def get_weights(atomic_symbols, symmetrical_atoms):
    weights = [ 1.0 if symbol in relevant_elements else 0.0 for symbol in atomic_symbols ]
    weights = np.array(weights)
    for l in symmetrical_atoms:
        weight = 1.0/len(l)
        for i in l:
            weights[i] = weight
    return weights

# mean-squared loss (not RMS!)
def loss_function(output, data):
    predictions = output
    observations = data.y
    weights = data.weights
    normalization = weights.sum()
    residuals = (predictions-observations)
    loss = residuals.square() * weights
    loss = loss.sum() / normalization
    return loss, residuals

# represents a molecule and all its jiggled training examples 
class Molecule():
    def __init__(self, name,
                 atomic_symbols,
                 symmetrical_atoms,        # list of lists of 0-indexed atom numbers
                 perturbed_geometries,
                 perturbed_shieldings):
        self.name = name                                       # name of molecule
        self.atomic_symbols = atomic_symbols                   # vector of strings of length n_atoms
        self.n_atoms = len(atomic_symbols)                     # number of atoms
        self.perturbed_geometries = perturbed_geometries       # (n_examples, n_atoms, 3)

        # zero out shieldings for irrelevant atoms
        perturbed_shieldings = self.scaling_function(perturbed_shieldings)
        for i,a in enumerate(atomic_symbols):          
            if a not in relevant_elements:          
                perturbed_shieldings[:,i]=0.0
        self.perturbed_shieldings = perturbed_shieldings                # (n_examples, n_atoms, 1)
        self.features = get_one_hots(atomic_symbols)                    # (n_atoms, n_elements)
        self.weights = get_weights(atomic_symbols, symmetrical_atoms)   # (n_atoms,)

def str2array(s):
    # https://stackoverflow.com/questions/35612235/how-to-read-numpy-2d-array-from-string
    s = re.sub('\[ +', '[', s.strip())
    s = re.sub('[,\s]+', ', ', s)
    a = ast.literal_eval(s)
    if len(a) == 0 or a is None:
        return []
    else:
        for i, b in enumerate(a):
            for j, _ in enumerate(b):
                 a[i][j] += -1
        return a

# objects that be placed in a queue to trigger special concurrent behavior
class DatasetSignal(Enum):
    # if received, the process will stop gracefully
    STOP = 1

    def __str__(self):
        return self._name_

# process that reads an hdf5 file and returns Molecules
class DatasetReader(Process):
    def __init__(self, name, molecule_queue, hdf5_filenames,
                 n_consumers, examples_to_process):
        super().__init__(group=None, target=None, name=name)
        self.molecule_queue = molecule_queue   # where to place the generated Molecule objects
        self.hdf5_filenames = hdf5_filenames   # hdf5 files to process
        self.n_consumers = n_consumers         # this many STOP signals will be sent when finished
        self.examples_to_process = examples_to_process  # max number of examples to process
        self.examples_processed = 0            # how many examples have been processed so far
        self.hdf5_file_list_index = 0          # which hdf5 file
        self.hdf5_file_index = 0               # which example within the hdf5 file

    # process the data in all hdf5 files
    def run(self):
        # iterate through hdf5 filenames, picking up where we left off
        while self.hdf5_file_list_index < len(self.hdf5_filenames):
            hdf5_filename = self.hdf5_filenames[self.hdf5_file_list_index]
            should_break = self.read_hdf5(hdf5_filename)       
            if should_break:
                break
            self.hdf5_file_list_index += 1
            self.hdf5_file_index = 0

        # tell this many downstream consumers to shut down
        for i in range(self.n_consumers):
            self.molecule_queue.put(DatasetSignal.STOP)

    # process the data in this hdf5 file
    def read_hdf5(self, filename):
        with h5py.File(filename, "r") as h5:
            h5_keys = []
            h5_values = []
            for key in h5.keys:
                if not key.startswith("data_"):
                    continue
                value = np.array(h5[key])
                h5.keys.append(key)
                h5_values.append(value)

            while self.hdf5_file_index < len(h5_keys):
                dataset_name = h5_keys[self.hdf5_file_index]
                geometries_and_shieldings = h5_values[self.hdf5_file_index]
                assert np.shape(geometries_and_shieldings)[2] == 4, "should be x,y,z,shielding"
                
                dataset_number = dataset_name.split("_")[1]
                
                perturbed_geometries = geometries_and_shieldings[[0],:,:3]
                perturbed_shieldings = geometries_and_shieldings[[0],:,3]
                n_examples = np.shape(perturbed_geometries)[0]                
                n_atoms = np.shape(perturbed_geometries)[1]

                atomic_symbols = h5.attrs[f"atomic_symbols_{dataset_number}"]
                assert len(atomic_symbols) == n_atoms, \
                       f"expected {n_atoms} atomic_symbols, but got {len(atomic_symbols)}"
                for a in atomic_symbols:
                    assert a in all_elements, \
                    f"unexpected element!  need to add {a} to all_elements"
                
                symmetrical_atoms = str2array(h5.attrs[f"symmetrical_atoms_{dataset_number}"])

                # store the results
                molecule = Molecule(dataset_number, atomic_symbols, symmetrical_atoms,
                                    perturbed_geometries, perturbed_shieldings)
                self.molecule_queue.put(molecule)

                # update counters
                self.examples_processed += n_examples
                self.hdf5_file_index += 1

                # check whether we have processed enough
                if self.examples_processed == self.examples_to_process:
                    break

    # switch from test to train, keeping track of which examples
    # have already been processed (just by not resetting hdf5_file_list_index
    # and hdf5_file_index)
    def reset_counters(self, examples_to_process):
        self.examples_to_process = examples_to_process
        self.examples_processed = 0

# process that takes Molecules from a queue (molecule_queue) and places
# DataNeighbors into another queue (data_neighbors_queue)
class MoleculeProcessor(Process):
    def __init__(self, name, molecule_queue, data_neighbors_queue, max_radius, Rs_in, Rs_out):
        super().__init__(group=None, target=None, name=name)
        self.molecule_queue = molecule_queue
        self.data_neighbors_queue = data_neighbors_queue
        self.max_radius = max_radius   # for building adjacency graph
        self.Rs_in = Rs_in
        self.Rs_out = Rs_out

    def run(self):
        while True:
            work = self.molecule_queue.get()
            if work == DataSignal.STOP:
                data_neighbors_queue.put(DataSignal.STOP)
                break
            assert isinstance(work, Molecule), f"expected Molecule but got {type(work)} instead!"    
            data_neighbors = preprocess_molecule(work, data_neighbors_queue)
            for dn in data_neighbors:
                self.molecule_queue.put(dn)

    def preprocess_molecule(self, molecule):
        data_neighbors = []

        features = torch.tensor(molecule.features, dtype=torch.float64)
        weights = torch.tensor(molecule.weights, dtype=torch.float64)
        n_examples = len(molecule.perturbed_geometries)
        for j in range(n_examples):
            g = torch.tensor(molecule.perturbed_geometries[j,:,:], dtype=torch.float64)
            s = torch.tensor(molecule.perturbed_shieldings[j], dtype=torch.float64).unsqueeze(-1)  # [1,N]
            dn = dh.DataNeighbors(x=features, Rs_in=self.Rs_in, pos=g, r_max=self.max_radius,
                                  self_interaction=True, name=molecule.name,
                                  weights=weights, y=s, Rs_out = self.Rs_out)
            data_neighbors.add(dn)
        return data_neighbors
