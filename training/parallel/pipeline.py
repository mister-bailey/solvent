from enum import Enum
import re
import math
import os
import ast
from torch.multiprocessing import Process, Lock, Semaphore, Value, Queue, Manager, Pool
import time
import numpy as np
import h5py
import torch
torch.set_default_dtype(torch.float64)
import torch_geometric as tg
import e3nn
import e3nn.point.data_helpers as dh
import training_config
from molecule_pipeline import MoleculePipeline
import itertools

### Code to Generate Molecules ###

# all expected elements
all_elements = training_config.all_elements
n_elements = len(all_elements)

# so we can normalize training data for the nuclei to be predicted
relevant_elements = training_config.relevant_elements

# cpu or gpu
device = training_config.device

# other parameters
n_norm = training_config.n_norm
testing_size = training_config.testing_size

### Code for Storing Training Data ###

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

### Parallel Preprocessing Code ###

# polls a lock/semaphore without acquiring it
# returns: True if it was positive, False otherwise
def check_semaphore(s):
    if s.acquire(False):
        s.release()
        return True
    return False

def set_semaphore(s, x):
    if x and not check_semaphore(s):
        s.release()
    elif (not x) and check_semaphore(s):
        s.acquire()

def wait_semaphore(s):
    s.acquire()
    s.release()

# returns total width (in floating point numbers) of Rs data
def Rs_size(Rs):
    size = 0
    for mul, l, _ in Rs:
        size += mul * (2 * l + 1)
    #print(f"Computed Rs_size = {size}")
    return size

class Pipeline():
    def __init__(self, hdf5_filenames, batch_size, max_radius, Rs_in, Rs_out, requested_jiggles=1,
                n_molecule_processors=1, molecule_cap = 10000, example_cap = 10000, batch_cap = 100,
                share_batches=True, manager=None, new_process=True):
        if manager is None:
            manager = Manager()
        #self.manager = manager
        self.knows = Semaphore(0) # > 0 if we know if any are coming
        self.working = Semaphore(1) # == 0 if DatasetReader is processing a command
        self.finished_reading = Lock() # locked if we're still reading from file
        self.in_pipe = Value('i',0) # number of molecules that have been sent to the pool

        self.command_queue = manager.Queue(10)
        #self.molecule_queue = manager.Queue(max_size)
        self.molecule_pipeline = None
        self.batch_queue = manager.Queue(batch_cap)
        self.share_batches = share_batches
        #self.molecule_processor_pool = Pool(n_molecule_processors, process_molecule,
        #                                    (self, max_radius, Rs_in, Rs_out))
        self.testing_molecules_dict = manager.dict()
        self.dataset_reader = DatasetReader("dataset_reader", self, hdf5_filenames, batch_size,
                    max_radius, Rs_in, Rs_out, requested_jiggles, n_molecule_processors, molecule_cap, example_cap,
                    batch_cap, self.testing_molecules_dict, new_process)
        if new_process:
            self.dataset_reader.start()

    #def __getstate__(self):
    #    self_dict = self.__dict__.copy()
    #    self_dict['molecule_processor_pool'] = None
    #    return self_dict

    # methods for pipeline user/consumer:
    def start_reading(self, examples_to_read, make_molecules=True, record_in_dict=False, batch_size=None, wait=False):
        #print("Start reading...")
        assert check_semaphore(self.finished_reading), "Tried to start reading file, but already reading!"
        with self.in_pipe.get_lock():
            assert self.in_pipe.value == 0, "Tried to start reading, but examples already in pipe!"
        set_semaphore(self.finished_reading, False)
        set_semaphore(self.knows, False)
        self.working.acquire()
        self.command_queue.put(StartSignal(examples_to_read, make_molecules, record_in_dict, batch_size))
        if wait: self.wait_till_done()

    def wait_till_done(self):
        #wait_semaphore(self.knows)
        #wait_semaphore(self.finished_reading)
        self.working.acquire()
        self.working.release()
        if self.any_coming():
            with self.in_pipe.get_lock():
                ip = self.in_pipe.value
            raise Exception(f"Waiting with {ip} examples in pipe!")

    def restart(self):
        assert check_semaphore(self.knows), "Tried to restart, but don't know if finished!"
        assert check_semaphore(self.finished_reading), "Tried to restart, but not finished reading!"
        assert not self.any_coming() , "Tried to restart pipeline before it was empty!"
        self.working.acquire()
        self.command_queue.put(RestartSignal())
        # What to do if things are still in the pipe???

    def any_coming(self): # returns True if at least one example is coming
        wait_semaphore(self.knows)
        with self.in_pipe.get_lock():
            return self.in_pipe.value > 0

    def get_batch(self, timeout=None):
        assert self.any_coming(), "Tried to get data from an empty pipeline!"
        x = self.batch_queue.get(True, timeout)
        with self.in_pipe.get_lock():
            self.in_pipe.value -= x.n_examples
            if self.in_pipe.value == 0 and not check_semaphore(self.finished_reading):
                set_semaphore(self.knows, False)
        return x

    def close(self):
        self.dataset_reader.terminate()
        self.dataset_reader.join()

    # methods for DatasetReader:
    def get_command(self):
        return self.command_queue.get()

    def put_molecule_to_ext(self, m, block=True):
        r = self.molecule_pipeline.put_molecule(m, block)
        if not r:
            return False
        with self.in_pipe.get_lock():
            if self.in_pipe.value == 0:
                set_semaphore(self.knows, True)
            self.in_pipe.value += m.perturbed_geometries.shape[0]
        return True

    def get_batch_from_ext(self, block=True):
        return self.molecule_pipeline.get_next_batch(block)

    def ext_batch_ready(self):
        return self.molecule_pipeline.batch_ready()
            
    def set_finished_reading(self): # !!! Call only after you've put the molecules !!!
        #print("*** Finished reading ***")
        set_semaphore(self.finished_reading, True)
        set_semaphore(self.knows, True)
        self.molecule_pipeline.notify_finished()
        #print(f"*** self.knows = {check_semaphore(self.knows)} ***")

    def put_batch(self, x):
        #self.batch_queue.put(DummyBatch(x.n_examples))
        #return
        if self.share_batches:
            x.share_memory_()
        self.batch_queue.put(x)

class DummyBatch():
    def __init__(self, n_examples):
        self.n_examples = n_examples

class DatasetSignal():
    def __str__(self):
        return "DatasetSignal"

class RestartSignal(DatasetSignal):
    def __str__(self):
        return "RestartSignal"

class StartSignal(DatasetSignal):
    def __init__(self, examples_to_read, make_molecules=True, record_in_dict=False, batch_size=None):
        self.examples_to_read = examples_to_read
        self.make_molecules = make_molecules
        self.record_in_dict = record_in_dict
        self.batch_size = batch_size

    def __str__(self):
        r = f"StartSignal(examples_to_read={self.examples_to_read}, make_molecules={self.make_molecules}, record_in_dict={self.record_in_dict}"
        if self.batch_size is not None:
            r += f", batch_size={self.batch_size}"
        return r + ")"

# process that reads an hdf5 file and returns Molecules
class DatasetReader(Process):
    def __init__(self, name, pipeline,
                       hdf5_filenames, batch_size, max_radius, Rs_in, Rs_out, requested_jiggles,
                       num_threads, molecule_cap, example_cap, batch_cap,
                       testing_molecules_dict, new_process = True):
        if new_process:
            super().__init__(group=None, target=None, name=name)
        self.pipeline = pipeline
        self.hdf5_filenames = hdf5_filenames   # hdf5 files to process
        self.hdf5_file_list_index = 0          # which hdf5 file
        self.hdf5_file_index = 0               # which example within the hdf5 file
        assert isinstance(requested_jiggles, int), "requested_jiggles should be an integer"
        self.requested_jiggles = requested_jiggles  # how many jiggles per file
        self.testing_molecules_dict = testing_molecules_dict   # molecule name -> Molecule
        self.molecule_pipeline = None
        self.molecule_pipeline_args = (batch_size, max_radius, Rs_size(Rs_in),
                    Rs_size(Rs_out), num_threads, molecule_cap, example_cap, batch_cap)
        print("(batch_size, max_radius, x_size, y_size, num_threads, molecule_cap, example_cap, batch_cap)")
        print(f" = {self.molecule_pipeline_args}")
        self.batch_number = 0
        self.molecule_number = 0

    # process the data in all hdf5 files
    def run(self):
        if self.molecule_pipeline is None:
            self.molecule_pipeline = MoleculePipeline(*self.molecule_pipeline_args)
            self.pipeline.molecule_pipeline = self.molecule_pipeline
        assert len(self.hdf5_filenames) > 0, "no files to process!"

        while True:
            command = self.pipeline.get_command()
            #print(f"Command: {command}")
            if isinstance(command, RestartSignal):
                self.hdf5_file_list_index = 0
                self.hdf5_file_index = 0
                #self.molecule_pipeline.notify_starting()
            elif isinstance(command, StartSignal):
                self.batch_number = 0
                self.molecule_number = 0
                self.molecule_pipeline.notify_starting(command.batch_size)
                self.read_examples(command.examples_to_read, command.make_molecules, command.record_in_dict)
                self.pipeline.set_finished_reading()
            else:
                raise ValueError("unexpected work type")
            self.pipeline.working.release()

    # iterate through hdf5 filenames, picking up where we left off
    # returns: number of examples processed
    def read_examples(self, examples_to_read, make_molecules, record_in_dict):
        examples_read = 0       # how many examples have been processed this round
        assert self.hdf5_file_list_index < len(self.hdf5_filenames), \
            "request to read examples, but files are finished!"
        while examples_read < examples_to_read:
            hdf5_filename = self.hdf5_filenames[self.hdf5_file_list_index]
            #print(f"{self.name}: filename={hdf5_filename} file_list_index={self.hdf5_file_list_index} file_index={self.hdf5_file_index}")
            examples_read += self.read_hdf5(hdf5_filename, examples_to_read - examples_read, make_molecules, record_in_dict)
            if self.hdf5_file_list_index >= len(self.hdf5_filenames):
                break
        return examples_read

    # process the data in this hdf5 file
    # requested_jiggles examples will be taken: 1 (default) or listlike
    # make_molecules: boolean that tells us if we should make Molecules objects
    #                 or just skip over these records
    # returns: number of molecules read from this file
    def read_hdf5(self, filename, examples_to_read, make_molecules, record_in_dict):
        with h5py.File(filename, "r") as h5:
            h5_keys = (k for k in h5.keys() if k.startswith("data_"))

            examples_read = 0
            for dataset_name in itertools.islice(h5_keys, self.hdf5_file_index, None):
                n_examples = min(examples_to_read - examples_read, self.requested_jiggles, h5[dataset_name].shape[0])
                #print(f"Reading molecule {dataset_name}.")
                if make_molecules:
                    geometries_and_shieldings = h5[dataset_name][:n_examples]

                    #print("AAAAAAA")
                    assert np.shape(geometries_and_shieldings)[2] == 4, "should be x,y,z,shielding"

                    dataset_number = dataset_name.split("_")[1]

                    #if isinstance(requested_jiggles, int):
                    #    jiggles = list(range(requested_jiggles))
                    #elif isinstance(requested_jiggles, list):
                    #    jiggles = requested_jiggles

                    #if jiggles_needed < len(jiggles):
                    #    jiggles = jiggles[:jiggles_needed]

                    # NOTE: We don't split a single molecule between different batches!
                    perturbed_geometries = geometries_and_shieldings[:,:,:3]
                    perturbed_shieldings = geometries_and_shieldings[:,:,3]
                    n_atoms = perturbed_geometries.shape[1]

                    #print("BBBBBBB")

                    atomic_symbols = h5.attrs[f"atomic_symbols_{dataset_number}"]

                    #print("   BBBB")
                    assert len(atomic_symbols) == n_atoms, \
                        f"expected {n_atoms} atomic_symbols, but got {len(atomic_symbols)}"
                    for a in atomic_symbols:
                        assert a in all_elements, \
                        f"unexpected element!  need to add {a} to all_elements"

                    #print("CCCCCCC")

                    symmetrical_atoms = str2array(h5.attrs[f"symmetrical_atoms_{dataset_number}"])

                    #print("DDDDDDD")

                    # store the results
                    molecule = Molecule(dataset_number, atomic_symbols, symmetrical_atoms,
                                        perturbed_geometries, perturbed_shieldings)
                    #print(f"Putting molecule {self.molecule_number} to pipeline.")
                    self.pipeline.put_molecule_to_ext(molecule)
                    self.molecule_number += 1
                    if record_in_dict:
                        self.testing_molecules_dict[molecule.name] = molecule
                    #print(f"put molecule {molecule.name}, n_examples={n_examples}")

                    while self.pipeline.ext_batch_ready():
                        #print(f"Getting batch {self.batch_number} from extension...")
                        batch = self.pipeline.get_batch_from_ext()
                        #print(f"Putting batch {self.batch_number} to queue...", end='')
                        self.pipeline.put_batch(batch)
                        #print(" Put.")
                        self.batch_number += 1

                # update counters
                examples_read += n_examples
                self.hdf5_file_index += 1

                # check whether we have processed enough
                if examples_read == examples_to_read:
                    # read enough examples, stopped partway through file
                    #print(f"Read {examples_read} examples from file {self.hdf5_file_list_index}.")
                    self.pipeline.set_finished_reading()
                    while self.molecule_pipeline.any_batch_coming():
                        #time.sleep(2)
                        #print(f"Molecule queue: {self.molecule_pipeline.molecule_queue_size()}")
                        #print(f"Example queue: {self.molecule_pipeline.example_queue_size()}")
                        #print(f"Batch queue: {self.molecule_pipeline.batch_queue_size()}")
                        #print(f"Num example: {self.molecule_pipeline.num_example()}")
                        #print(f"Num batch: {self.molecule_pipeline.num_batch()}\n")
                        #print(f"Getting batch {self.batch_number} from extension...")
                        batch = self.pipeline.get_batch_from_ext()
                        #print(f"Putting batch {self.batch_number} to queue...", end='')
                        self.pipeline.put_batch(batch)
                        #print(" Put.")
                        self.batch_number += 1
                    return examples_read

        # reached end of file without enough examples
        #print(f"Finished reading file {self.hdf5_file_list_index} with {examples_read} examples.")
        self.hdf5_file_list_index += 1
        self.hdf5_file_index = 0
        return examples_read

# method that takes Molecules from a queue (molecule_queue) and places
# DataNeighbors into another queue (data_neighbors_queue)
def process_molecule(pipeline, max_radius, Rs_in, Rs_out):
    while True:
        molecule = pipeline.get_molecule()
        #print(f"> got molecule {molecule.name}, n_examples={len(molecule.perturbed_geometries)}")
        assert isinstance(molecule, Molecule), \
               f"expected Molecule but got {type(molecule)} instead!"

        features = torch.tensor(molecule.features, dtype=torch.float64)
        weights = torch.tensor(molecule.weights, dtype=torch.float64)
        n_examples = len(molecule.perturbed_geometries)
        for j in range(n_examples):
            g = torch.tensor(molecule.perturbed_geometries[j,:,:], dtype=torch.float64)
            s = torch.tensor(molecule.perturbed_shieldings[j], dtype=torch.float64).unsqueeze(-1)  # [1,N]
            dn = dh.DataNeighbors(x=features, Rs_in=Rs_in, pos=g, r_max=max_radius,
                                  self_interaction=True, name=molecule.name,
                                  weights=weights, y=s, Rs_out=Rs_out)
            pipeline.put_data_neighbor(dn)

position_tolerance = .00001
shielding_tolerance = .000001
# compares two different data neighbors structures (presumably generated in python and c++)
# confirmed: C++ pipeline produces equivalent results to DataNeighbors 
def compare_data_neighbors(dn1, dn2):
    print("Comparing pair of Data Neighbors structures...")
    if dn1.pos.shape[0] != dn2.pos.shape[0]:
        raise ValueError(f"Different numbers of atoms! {dn1.pos.shape[0]} vs {dn2.pos.shape[0]}")
    n_atoms = dn1.pos.shape[0]
    print(f"Comparing {n_atoms} atoms...")
    atom_map = [0] * n_atoms
    atom_taken = [False] * n_atoms
    for i in range(n_atoms):
        for j in range(n_atoms):
            if (not atom_taken[j]) and (torch.norm(dn1.pos[i,:] - dn2.pos[j,:]) <= position_tolerance):
                atom_map[i] = j
                atom_taken[j] = True
                if not torch.equal(dn1.x[i], dn2.x[j]):
                    print(f"1-hots don't match for atom {i}!")
                    raise ValueError()
                if abs(dn1.y[i] - dn2.y[j]) > shielding_tolerance:
                    print(f"Shieldings don't match for atom {j}! {dn1.y[i]} vs {dn2.y[j]}")
                    raise ValueError()
                break
        else:
            print(f"Could not match atom {i}!")
            raise ValueError()
    print(f"Matched {n_atoms} atoms.  atom_map: ", atom_map)

    if dn1.edge_attr.shape[0] != dn2.edge_attr.shape[0]:
        raise ValueError(f"Different numbers of edges! {dn1.edge_attr.shape[0]} vs {dn2.edge_attr.shape[0]}")
    n_edges = dn1.edge_attr.shape[0]
    print(f"Comparing {n_edges} edges...")
    edge_taken = [False] * n_edges
    for a in range(n_edges):
        e1 = torch.tensor([atom_map[dn1.edge_index[0,a]], atom_map[dn1.edge_index[1,a]]])
        for b in range(n_edges):
            if edge_taken[b]: continue
            e2 = dn2.edge_index[:,b]
            if torch.equal(e1, e2):
                if torch.norm(dn1.edge_attr[a,:] - dn2.edge_attr[b,:]) > position_tolerance:
                    print(f"Vectors don't match for edges {a} and {b} : ({dn1.edge_index[0,a]}) -> ({dn1.edge_index[1,a]})")
                    print(f"{dn1.edge_attr[a,:]} vs { dn2.edge_attr[b,:]}")
                    raise ValueError()
                edge_taken[b] = True
                break
        else:
            print(f"Could not match edge {a}!")
            raise ValueError()
    print(f"Matched {n_edges} edges.")

    print("Data Neighbors matched!")

def test_data_neighbors(example, Rs_in, Rs_out, max_radius, molecule_dict):
    dn1 = example
    molecule = molecule_dict[dn1.name]
    features = torch.tensor(molecule.features, dtype=torch.float64)
    weights = torch.tensor(molecule.weights, dtype=torch.float64)
    g = torch.tensor(molecule.perturbed_geometries[0,:,:], dtype=torch.float64)
    s = torch.tensor(molecule.perturbed_shieldings[0], dtype=torch.float64).unsqueeze(-1)  # [1,N]
    dn2 = dh.DataNeighbors(x=features, Rs_in=Rs_in, pos=g, r_max=max_radius,
            self_interaction=True, name=molecule.name, weights=weights, y=s, Rs_out=Rs_out)
    compare_data_neighbors(dn1, dn2)


