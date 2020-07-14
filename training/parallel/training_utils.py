from enum import Enum
import re
import ast
from multiprocessing import Process, Lock, Semaphore, Value
import numpy as np
import h5py
import torch
torch.set_default_dtype(torch.float64)
import torch_geometric as tg
import e3nn
import e3nn.point.data_helpers as dh
import training_config

# all expected elements
all_elements = training_config.all_elements
n_elements = len(all_elements)

# so we can normalize training data for the nuclei to be predicted
relevant_elements = training_config.relevant_elements

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

# polls a lock/semaphore without acquiring it
# returns: True if it was locked, False otherwise
def locked(lock):
    if lock.acquire(False):
        lock.release()
        return False
    return True

def set_semaphore(s, x):
    if x and locked(s):
        s.release()
    elif (not x) and (not locked(s)):
        s.acquire()

class PipelineReporter():
    def __init__(self, manager=None, active=True):
        if manager is not None:
            self.reporting = manager.Semaphore(0)
            self.active = manager.Value('i',int(active))
            self.in_pipe = manager.Value('i',0)
        else:
            self.reporting = Semaphore(0)
            self.active = Value('i',int(active))
            self.in_pipe = Value('i',0)
        
    def set_reporting(self, reporting=True):
        set_semaphore(self.reporting, reporting)

    def set_active(self):
        with self.active.get_lock():
            self.active.value = True
        self.set_reporting()
    
    def set_inactive(self):
        with self.active.get_lock():
            self.active.value = False
        self.set_reporting()

    def get_report(self):
        self.reporting.acquire()
        self.reporting.release()
        with self.active.get_lock():
            return self.active.value

    def add_to_pipe(self, n=1):
        with self.in_pipe.get_lock():
            self.in_pipe.value += n
    
    def take_from_pipe(self, n=1):
        self.add_to_pipe(-n)

    def num_in_pipe(self):
        with self.in_pipe.get_lock():
            return self.in_pipe.value

    def any_in_pipe(self):
        return self.num_in_pipe() > 0

    def any_coming(self):
        return self.get_report() or self.any_in_pipe()


    


class DatasetSignal(Enum):
    # a "poison pill" that signals to the final thread
    # that a phase of calculation is complete
    STOP = 1

    # tells process_datasets to start from the beginning of the file
    RESTART = 2

    def __str__(self):
        return self._name_

# process that reads an hdf5 file and returns Molecules
class DatasetReader(Process):
    def __init__(self, name, example_queue, molecule_queue, pipeline_reporter,
                       hdf5_filenames):
        super().__init__(group=None, target=None, name=name)
        self.example_queue = example_queue     # where to get work from
        self.molecule_queue = molecule_queue   # where to place the generated Molecule objects
        self.pipeline_reporter = pipeline_reporter
        self.hdf5_filenames = hdf5_filenames   # hdf5 files to process
        self.hdf5_file_list_index = 0          # which hdf5 file
        self.hdf5_file_index = 0               # which example within the hdf5 file

    # process the data in all hdf5 files
    def run(self):
        assert len(self.hdf5_filenames) > 0, "no files to process!"

        while True:
            work = self.example_queue.get()
            if work == DatasetSignal.RESTART:
                self.hdf5_file_list_index = 0
                self.hdf5_file_index = 0
                self.pipeline_reporter.set_active()
                continue
            elif work == DatasetSignal.STOP:
                break
            elif isinstance(work, tuple):
                assert len(work) == 2, \
                       "expected 2-tuple: (examples_to_process, make_molecules)"
                examples_to_process, make_molecules = work
                self.read_examples(examples_to_process, make_molecules)
                self.pipeline_reporter.set_inactive()
            else:
                raise ValueError("unexpected work type")

    # iterate through hdf5 filenames, picking up where we left off
    # returns: number of examples processed
    def read_examples(self, examples_to_read, make_molecules=True, requested_jiggles=1):
        examples_processed = 0       # how many examples have been processed this round
        assert self.hdf5_file_list_index < len(self.hdf5_filenames), \
            "request to read examples, but files are finished!"
        while examples_processed < examples_to_read:
            hdf5_filename = self.hdf5_filenames[self.hdf5_file_list_index]
            print(f"{self.name}: filename={hdf5_filename} file_list_index={self.hdf5_file_list_index} file_index={self.hdf5_file_index}")
            examples_processed += self.read_hdf5(hdf5_filename, examples_to_read - examples_processed, make_molecules, requested_jiggles)
            if self.hdf5_file_list_index >= len(self.hdf5_filenames):
                self.pipeline_reporter.set_inactive()
                break
                    
        return examples_processed



    # process the data in this hdf5 file
    # requested_jiggles examples will be taken: 1 (default) or listlike
    # make_molecules: boolean that tells us if we should make Molecules objects
    #                 or just skip over these records
    # returns: number of molecules read from this file
    def read_hdf5(self, filename, examples_to_read, make_molecules, requested_jiggles=1):
        with h5py.File(filename, "r") as h5:
            h5_keys = []
            h5_values = []
            for key in h5.keys():
                if not key.startswith("data_"):
                    continue
                value = np.array(h5[key])
                h5_keys.append(key)
                h5_values.append(value)

            examples_read = 0
            while examples_read < examples_to_read and self.hdf5_file_index < len(h5_keys):
                dataset_name = h5_keys[self.hdf5_file_index]
                geometries_and_shieldings = h5_values[self.hdf5_file_index]
                assert np.shape(geometries_and_shieldings)[2] == 4, "should be x,y,z,shielding"

                dataset_number = dataset_name.split("_")[1]

                if isinstance(requested_jiggles, int):
                    jiggles = range(requested_jiggles) #[ i for i in range(requested_jiggles) ]
                elif isinstance(requested_jiggles, list):
                    jiggles = requested_jiggles

                jiggles_needed = examples_to_read - examples_read
                assert jiggles_needed >= 1
                if jiggles_needed < len(jiggles):
                    jiggles = jiggles[:jiggles_needed]
                # NOTE: We don't split a single molecule between different batches!

                perturbed_geometries = geometries_and_shieldings[jiggles,:,:3]
                perturbed_shieldings = geometries_and_shieldings[jiggles,:,3]
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
                if make_molecules:
                    molecule = Molecule(dataset_number, atomic_symbols, symmetrical_atoms,
                                        perturbed_geometries, perturbed_shieldings)
                    self.molecule_queue.put(molecule)
                    self.pipeline_reporter.set_active()
                    self.pipeline_reporter.add_to_pipe(n_examples)

                # update counters
                examples_read += n_examples
                #print(f"{self.name} has processed {self.examples_processed} examples")
                self.hdf5_file_index += 1

                # check whether we have processed enough
                assert examples_read <= examples_to_read,\
                       f"have processed {examples_read} examples but {examples_to_read} examples were requested"
                if examples_read == examples_to_read:
                    # we should break
                    return examples_read
            if self.hdf5_file_index >= len(h5_keys):
                self.hdf5_file_list_index += 1
                self.hdf5_file_index = 0



        # we didn't reach the desired number of examples
        return examples_read

# method that takes Molecules from a queue (molecule_queue) and places
# DataNeighbors into another queue (data_neighbors_queue)
def process_molecule(molecule_queue, data_neighbors_queue, \
                     max_radius, Rs_in, Rs_out):
    while True:
        work = molecule_queue.get()
        if work == DatasetSignal.STOP:
            #data_neighbors_queue.put(DatasetSignal.STOP)
            continue
        assert isinstance(work, Molecule), \
               f"expected Molecule but got {type(work)} instead!"

        data_neighbors = []
        molecule = work
        features = torch.tensor(molecule.features, dtype=torch.float64)
        weights = torch.tensor(molecule.weights, dtype=torch.float64)
        n_examples = len(molecule.perturbed_geometries)
        for j in range(n_examples):
            g = torch.tensor(molecule.perturbed_geometries[j,:,:], dtype=torch.float64)
            s = torch.tensor(molecule.perturbed_shieldings[j], dtype=torch.float64).unsqueeze(-1)  # [1,N]
            dn = dh.DataNeighbors(x=features, Rs_in=Rs_in, pos=g, r_max=max_radius,
                                  self_interaction=True, name=molecule.name,
                                  weights=weights, y=s, Rs_out = Rs_out)
            data_neighbors.append(dn)

        for dn in data_neighbors:
            data_neighbors_queue.put(dn)
