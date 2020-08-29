import itertools
from molecule_pipeline import MoleculePipeline
#import training_config
import e3nn.point.data_helpers as dh
import e3nn
import torch_geometric as tg
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

### Code to Generate Molecules ###

#symbol_to_number = training_config.symbol_to_number
#number_to_symbol = training_config.number_to_symbol

# all expected elements
#all_elements = training_config.all_elements
#n_elements = len(all_elements)

# so we can normalize training data for the nuclei to be predicted
#relevant_elements = training_config.relevant_elements

# cpu or gpu
#device = training_config.device

# other parameters
#testing_size = training_config.testing_size

### Code for Storing Training Data ###

# represents a molecule and all its jiggled training examples


class Molecule():
    def __init__(self, name,
                 perturbed_geometries,
                 perturbed_shieldings,
                 atomic_numbers,
                 symmetrical_atoms=None,        # list of lists of 0-indexed atom numbers
                 weights=None):
        self.name = name                                       # name of molecule
        # vector of strings of length n_atoms
        self.atomic_numbers = atomic_numbers
        # number of atoms
        self.n_atoms = len(atomic_numbers)
        # (n_examples, n_atoms, 3)
        self.perturbed_geometries = perturbed_geometries

        # zero out shieldings for irrelevant atoms
        # for i,a in enumerate(atomic_numbers):
        #    #if isinstance(a, str): a = symbol_to_number[a]
        #    if a not in training_config.relevant_elements:
        #        perturbed_shieldings[...,i]=0.0  # (n_examples, n_atoms, 1)
        # (n_atoms, n_elements)
        self.perturbed_shieldings = perturbed_shieldings
        self.features = Molecule.get_one_hots(atomic_numbers)
        if weights is None:
            self.weights = Molecule.get_weights(
                atomic_numbers, symmetrical_atoms, None)   # (n_atoms,)
        else:
            self.weights = weights

    one_hot_table = np.zeros((0,0))

    @staticmethod
    def initialize_one_hot_table(all_elements):
        max_element = max(all_elements)
        Molecule.one_hot_table = np.zeros((max_element+1, len(all_elements)), dtype=np.float64)
        for i, e in enumerate(all_elements):
            Molecule.one_hot_table[e][i] = 1.0

    # generates one-hots for a list of atomic_symbols
    @staticmethod
    def get_one_hots(atomic_numbers):
        return Molecule.one_hot_table[atomic_numbers]

    # compute weights for loss function
    @staticmethod
    def get_weights(atomic_symbols, symmetrical_atoms, relevant_elements):
        weights = [
            1.0 if symbol in relevant_elements else 0.0 for symbol in atomic_symbols]
        weights = np.array(weights)
        for l in symmetrical_atoms:
            weight = 1.0/len(l)
            for i in l:
                weights[i] = weight
        return weights


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
    def __init__(self, config, share_batches=True, manager=None, new_process=True):
        if manager is None:
            manager = Manager()
        #self.manager = manager
        self.knows = Semaphore(0)  # > 0 if we know if any are coming
        # == 0 if DatasetReader is processing a command
        self.working = Semaphore(1)
        self.finished_reading = Lock()  # locked if we're still reading from file
        # number of molecules that have been sent to the pool
        self.in_pipe = Value('i', 0)

        self.command_queue = manager.Queue(10)
        #self.molecule_queue = manager.Queue(max_size)
        self.molecule_pipeline = None
        self.batch_queue = manager.Queue(config.batch_queue_cap)
        self.share_batches = share_batches
        # self.molecule_processor_pool = Pool(n_molecule_processors, process_molecule,
        #                                    (self, max_radius, Rs_in, Rs_out))
        self.testing_molecules_dict = manager.dict()

        self.dataset_reader = DatasetReader("dataset_reader", self, config, self.testing_molecules_dict, new_process)
        if new_process:
            self.dataset_reader.start()

    # def __getstate__(self):
    #    self_dict = self.__dict__.copy()
    #    self_dict['molecule_processor_pool'] = None
    #    return self_dict

    # methods for pipeline user/consumer:
    def start_reading(self, examples_to_read, make_molecules=True, record_in_dict=False, batch_size=None, wait=False):
        #print("Start reading...")
        assert check_semaphore(
            self.finished_reading), "Tried to start reading file, but already reading!"
        with self.in_pipe.get_lock():
            assert self.in_pipe.value == 0, "Tried to start reading, but examples already in pipe!"
        set_semaphore(self.finished_reading, False)
        set_semaphore(self.knows, False)
        self.working.acquire()
        self.command_queue.put(StartReading(
            examples_to_read, make_molecules, record_in_dict, batch_size))
        if wait:
            self.wait_till_done()

    def wait_till_done(self):
        # wait_semaphore(self.knows)
        # wait_semaphore(self.finished_reading)
        self.working.acquire()
        self.working.release()
        if self.any_coming():
            with self.in_pipe.get_lock():
                ip = self.in_pipe.value
            raise Exception(f"Waiting with {ip} examples in pipe!")

    def restart(self):
        assert check_semaphore(
            self.knows), "Tried to restart, but don't know if finished!"
        assert check_semaphore(
            self.finished_reading), "Tried to restart, but not finished reading!"
        assert not self.any_coming(), "Tried to restart pipeline before it was empty!"
        self.working.acquire()
        self.command_queue.put(RestartSignal())
        # What to do if things are still in the pipe???

    def set_indices(self, test_set_indices):
        self.working.acquire()
        self.command_queue.put(SetIndices(test_set_indices))
        self.working.acquire()
        self.command_queue.put(RestartSignal())

    #def read_test_set(self, record_in_dict=True, batch_size=1):
    #    #print("Start reading...")
    #    assert check_semaphore(
    #        self.finished_reading), "Tried to start reading file, but already reading!"
    #    with self.in_pipe.get_lock():
    #        assert self.in_pipe.value == 0, "Tried to start reading, but examples already in pipe!"
    #    set_semaphore(self.finished_reading, False)
    #    set_semaphore(self.knows, False)
    #    self.working.acquire()
    #    self.command_queue.put(ReadTestSet(record_in_dict, batch_size))

    def any_coming(self):  # returns True if at least one example is coming
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
            if m.perturbed_geometries.ndim == 3:
                self.in_pipe.value += m.perturbed_geometries.shape[0]
            else:
                self.in_pipe.value += 1
        return True

    def put_molecule_data(self, data, atomic_numbers, weights, name, block=True):
        r = self.molecule_pipeline.put_molecule_data(
            data, atomic_numbers, weights, name, block)
        if not r:
            return False
        with self.in_pipe.get_lock():
            if self.in_pipe.value == 0:
                set_semaphore(self.knows, True)
            if data.ndim == 3:
                self.in_pipe.value += data.shape[0]
            else:
                self.in_pipe.value += 1
        return True

    def get_batch_from_ext(self, block=True):
        return self.molecule_pipeline.get_next_batch(block)

    def ext_batch_ready(self):
        return self.molecule_pipeline.batch_ready()

    # !!! Call only after you've put the molecules !!!
    def set_finished_reading(self):
        #print("*** Finished reading ***")
        set_semaphore(self.finished_reading, True)
        set_semaphore(self.knows, True)
        self.molecule_pipeline.notify_finished()
        #print(f"*** self.knows = {check_semaphore(self.knows)} ***")

    def put_batch(self, x):
        if self.share_batches:
            x.share_memory_()
        self.batch_queue.put(x)


class DatasetSignal():
    def __str__(self):
        return "DatasetSignal"


class RestartSignal(DatasetSignal):
    def __str__(self):
        return "RestartSignal"


class StartReading(DatasetSignal):
    def __init__(self, examples_to_read, make_molecules=True, record_in_dict=False, batch_size=None):
        self.examples_to_read = examples_to_read
        self.make_molecules = make_molecules
        self.record_in_dict = record_in_dict
        self.batch_size = batch_size

    def __str__(self):
        r = f"StartReading(examples_to_read={self.examples_to_read}, make_molecules={self.make_molecules}, record_in_dict={self.record_in_dict}"
        if self.batch_size is not None:
            r += f", batch_size={self.batch_size}"
        return r + ")"


class SetIndices(DatasetSignal):
    def __init__(self, indices):
        """
        indices should be sorted!!!
        """
        self.indices = indices

    def __str__(self):
        return f"SetIndices({len(self.indices)})"


#class ReadTestSet(DatasetSignal):
#    def __init__(self, record_in_dict=True, batch_size=1):
#        self.record_in_dict = record_in_dict
#        self.batch_size = batch_size
#
#    def __str__(self):
#        return f"ReadTestSet(record_in_dict={self.record_in_dict}, batch_size={self.batch_size})"

# process that reads an hdf5 file and returns Molecules


class DatasetReader(Process):
    def __init__(self, name, pipeline, config,
                 testing_molecules_dict, shuffle_incoming=True, new_process=True, requested_jiggles=1):
        if new_process:
            super().__init__(group=None, target=None, name=name)
        self.pipeline = pipeline
        self.config = config

        self.hdf5_file_list_index = 0          # which hdf5 file
        self.hdf5_file_index = 0               # which example within the hdf5 file
        self.all_elements = config.all_elements
        self.requested_jiggles = requested_jiggles  # how many jiggles per file
        self.testing_molecules_dict = testing_molecules_dict   # molecule name -> Molecule
        self.molecule_pipeline = None
        self.molecule_pipeline_args = (config.batch_size, config.max_radius, config.all_elements,
                                       config.relevant_elements, config.n_molecule_processors,
                                       config.molecule_queue_cap, config.example_queue_cap,
                                       config.batch_queue_cap)
        #self.molecule_number = 0
        self.index_pos = 0
        self.shuffle_incoming = shuffle_incoming

        self.data_source = config.data_source
        if self.data_source == 'hdf5':
            self.hdf5_filenames = config.hdf5_filenames   # hdf5 files to process
            self.read_examples = self.read_examples_from_file
            if config.file_format == 0:
                self.read_hdf5 = self.read_hdf5_format_0
            elif config.file_format == 1:
                self.read_hdf5 = self.read_hdf5_format_1
        elif self.data_source == 'SQL':
            self.connect_params = config.connect_params
            self.SQL_fetch_size = config.SQL_fetch_size
            self.molecule_buffer = []
            self.read_examples = self.read_examples_from_SQL
            from mysql_df import MysqlDB
            self.database = MysqlDB(self.connect_params)


    # process the data in all hdf5 files
    def run(self):
        if self.molecule_pipeline is None:
            self.molecule_pipeline = MoleculePipeline(*self.molecule_pipeline_args)
            self.pipeline.molecule_pipeline = self.molecule_pipeline
            self.indices = np.array([])
        if len(Molecule.one_hot_table) == 0:
            Molecule.initialize_one_hot_table(self.all_elements)

        while True:
            command = self.pipeline.get_command()
            #print(f"Command: {command}")
            if isinstance(command, RestartSignal):
                self.hdf5_file_list_index = 0
                self.hdf5_file_index = 0
                self.index_pos = 0
                #self.molecule_number = 0
                self.molecule_buffer = []
            elif isinstance(command, StartReading):
                self.molecule_pipeline.notify_starting(command.batch_size)
                self.read_examples(command.examples_to_read, command.make_molecules, command.record_in_dict)
                self.pipeline.set_finished_reading()
                while self.molecule_pipeline.any_batch_coming():
                    self.pipeline.put_batch(self.pipeline.get_batch_from_ext())
            elif isinstance(command, SetIndices):
                self.indices = command.indices
            else:
                raise ValueError("unexpected work type")
            self.pipeline.working.release()

    # iterate through hdf5 filenames, picking up where we left off
    # returns: number of examples processed
    def read_examples_from_file(self, examples_to_read, make_molecules, record_in_dict):
        examples_read = 0       # how many examples have been processed this round
        assert self.hdf5_file_list_index < len(self.hdf5_filenames), \
            "request to read examples, but files are finished!"
        while examples_read < examples_to_read:
            hdf5_filename = self.hdf5_filenames[self.hdf5_file_list_index]
            #print(f"{self.name}: filename={hdf5_filename} file_list_index={self.hdf5_file_list_index} file_index={self.hdf5_file_index}")
            examples_read += self.read_hdf5(
                hdf5_filename, examples_to_read - examples_read, make_molecules, record_in_dict)
            if self.hdf5_file_list_index >= len(self.hdf5_filenames):
                break
        return examples_read

    # process the data in this hdf5 file
    # requested_jiggles examples will be taken: 1 (default) or listlike
    # make_molecules: boolean that tells us if we should make Molecules objects
    #                 or just skip over these records
    # returns: number of molecules read from this file
    def read_hdf5_format_0(self, filename, examples_to_read, make_molecules, record_in_dict):
        with h5py.File(filename, "r") as h5:
            h5_keys = (k for k in h5.keys() if k.startswith("data_"))

            examples_read = 0
            for dataset_name in itertools.islice(h5_keys, self.hdf5_file_index, None):
                n_examples = min(examples_to_read - examples_read,
                                 self.requested_jiggles, h5[dataset_name].shape[0])
                if make_molecules:
                    geometries_and_shieldings = h5[dataset_name][:n_examples]

                    assert np.shape(geometries_and_shieldings)[
                        2] == 4, "should be x,y,z,shielding"

                    dataset_number = dataset_name.split("_")[1]

                    # NOTE: We don't split a single molecule between different batches!
                    perturbed_geometries = geometries_and_shieldings[:, :, :3]
                    perturbed_shieldings = geometries_and_shieldings[:, :, 3]
                    n_atoms = perturbed_geometries.shape[1]

                    atomic_symbols = h5.attrs[f"atomic_symbols_{dataset_number}"]

                    assert len(atomic_symbols) == n_atoms, \
                        f"expected {n_atoms} atomic_symbols, but got {len(atomic_symbols)}"
                    for a in atomic_symbols:
                        assert a in self.all_elements, \
                            f"unexpected element!  need to add {a} to all_elements"

                    symmetrical_atoms = str2array(
                        h5.attrs[f"symmetrical_atoms_{dataset_number}"])

                    # store the results
                    molecule = Molecule(dataset_number, perturbed_geometries, perturbed_shieldings,
                                        atomic_symbols, symmetrical_atoms=symmetrical_atoms)
                    self.pipeline.put_molecule_to_ext(molecule)
                    #self.molecule_number += 1
                    if record_in_dict:
                        self.testing_molecules_dict[molecule.name] = molecule

                    while self.pipeline.ext_batch_ready():
                        self.pipeline.put_batch(
                            self.pipeline.get_batch_from_ext())

                # update counters
                examples_read += n_examples
                self.hdf5_file_index += 1

                if examples_read == examples_to_read:
                    # read enough examples, stopped partway through file
                    self.pipeline.set_finished_reading()
                    return examples_read

        # reached end of file without enough examples
        self.hdf5_file_list_index += 1
        self.hdf5_file_index = 0
        return examples_read

    def read_hdf5_format_1(self, filename, examples_to_read, make_molecules, record_in_dict):
        with h5py.File(filename, "r") as h5:
            if make_molecules:
                examples_read = 0
                for key, dataset in itertools.islice(h5.items(), self.hdf5_file_index, None):
                    if record_in_dict:
                        molecule = Molecule(str(dataset.attrs["smiles"]), dataset[..., :3], dataset[..., 3],
                                            dataset.attrs["atomic_numbers"], weights=dataset.attrs["weights"])
                        self.pipeline.put_molecule_to_ext(molecule)
                        self.testing_molecules_dict[molecule.name] = molecule
                    else:
                        #print("Putting molecule data...")
                        # r = self.pipeline.put_molecule_data(dataset, dataset.attrs["atomic_numbers"],
                        #         dataset.attrs["weights"], str(dataset.attrs["smiles"]))
                        #print(f"Put molecule? {r}")
                        molecule = Molecule(str(dataset.attrs["smiles"]), dataset[..., :3], dataset[..., 3],
                                            dataset.attrs["atomic_numbers"], weights=dataset.attrs["weights"])
                        self.pipeline.put_molecule_to_ext(molecule)

                    while self.pipeline.ext_batch_ready():
                        self.pipeline.put_batch(
                            self.pipeline.get_batch_from_ext())

                    # update counters
                    examples_read += 1
                    self.hdf5_file_index += 1

                    if examples_read == examples_to_read:
                        # read enough examples, stopped partway through file
                        self.pipeline.set_finished_reading()
                        return examples_read

                # reached end of file without enough examples
                self.hdf5_file_list_index += 1
                self.hdf5_file_index = 0
                return examples_read
            else:
                file_length = len(h5.keys())
                if self.hdf5_file_index + examples_to_read >= file_length:
                    self.hdf5_file_list_index += 1
                    self.hdf5_file_index = 0
                    return file_length - self.hdf5_file_index
                else:
                    self.hdf5_file_index += examples_to_read
                    return examples_to_read

    def read_examples_from_SQL(self, examples_to_read, make_molecules, record_in_dict):
        examples_read = 0
        while examples_read < examples_to_read:
            for i, (_, data, weights, smiles) in enumerate(self.molecule_buffer):
                if make_molecules:
                    if examples_read == examples_to_read:
                        self.molecule_buffer = self.molecule_buffer[i:]
                        break
                    molecule = Molecule(str(smiles), data[:, 1:4], data[:, 4],
                                        data[:, 0].astype(np.int32), weights=weights)
                    self.pipeline.put_molecule_to_ext(molecule)
                    if record_in_dict:
                        self.testing_molecules_dict[molecule.name] = molecule
                examples_read += 1
                while self.pipeline.ext_batch_ready():
                    self.pipeline.put_batch(self.pipeline.get_batch_from_ext())
            if len(self.molecule_buffer) < self.SQL_fetch_size:
                self.molecule_buffer += self.database.read_rows(np.nditer(
                    self.indices[self.index_pos : self.index_pos + self.SQL_fetch_size]),
                    randomize = self.shuffle_incoming)
                self.index_pos += self.SQL_fetch_size

        return examples_read


# returns a random shuffle of the available indices, for test/train split and random training
def generate_index_shuffle(size, connect_params, rng=None, seed=None):
    from mysql_df import MysqlDB
    db = MysqlDB(connect_params)
    indices = np.array(db.get_finished_idxs(size), dtype=np.int32)
    if rng is None:
        rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return indices

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
            g = torch.tensor(
                molecule.perturbed_geometries[j, :, :], dtype=torch.float64)
            s = torch.tensor(
                molecule.perturbed_shieldings[j], dtype=torch.float64).unsqueeze(-1)  # [1,N]
            dn = dh.DataNeighbors(x=features, Rs_in=Rs_in, pos=g, r_max=max_radius,
                                  self_interaction=True, name=molecule.name,
                                  weights=weights, y=s, Rs_out=Rs_out)
            pipeline.put_data_neighbor(dn)


# compares two different data neighbors structures (presumably generated in python and c++)
# confirmed: C++ pipeline produces equivalent results to DataNeighbors
position_tolerance = .00001
shielding_tolerance = .000001


def compare_data_neighbors(dn1, dn2):
    print("Comparing pair of Data Neighbors structures...")
    if dn1.pos.shape[0] != dn2.pos.shape[0]:
        raise ValueError(
            f"Different numbers of atoms! {dn1.pos.shape[0]} vs {dn2.pos.shape[0]}")
    n_atoms = dn1.pos.shape[0]
    print(f"Comparing {n_atoms} atoms...")
    atom_map = [0] * n_atoms
    atom_taken = [False] * n_atoms
    for i in range(n_atoms):
        for j in range(n_atoms):
            if (not atom_taken[j]) and (torch.norm(dn1.pos[i, :] - dn2.pos[j, :]) <= position_tolerance):
                atom_map[i] = j
                atom_taken[j] = True
                if not torch.equal(dn1.x[i], dn2.x[j]):
                    print(f"1-hots don't match for atom {i}!")
                    raise ValueError()
                if abs(dn1.y[i] - dn2.y[j]) > shielding_tolerance:
                    print(
                        f"Shieldings don't match for atom {j}! {dn1.y[i]} vs {dn2.y[j]}")
                    raise ValueError()
                break
        else:
            print(f"Could not match atom {i}!")
            raise ValueError()
    print(f"Matched {n_atoms} atoms.  atom_map: ", atom_map)

    if dn1.edge_attr.shape[0] != dn2.edge_attr.shape[0]:
        raise ValueError(
            f"Different numbers of edges! {dn1.edge_attr.shape[0]} vs {dn2.edge_attr.shape[0]}")
    n_edges = dn1.edge_attr.shape[0]
    print(f"Comparing {n_edges} edges...")
    edge_taken = [False] * n_edges
    for a in range(n_edges):
        e1 = torch.tensor([atom_map[dn1.edge_index[0, a]],
                           atom_map[dn1.edge_index[1, a]]])
        for b in range(n_edges):
            if edge_taken[b]:
                continue
            e2 = dn2.edge_index[:, b]
            if torch.equal(e1, e2):
                if torch.norm(dn1.edge_attr[a, :] - dn2.edge_attr[b, :]) > position_tolerance:
                    print(
                        f"Vectors don't match for edges {a} and {b} : ({dn1.edge_index[0,a]}) -> ({dn1.edge_index[1,a]})")
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
    g = torch.tensor(molecule.perturbed_geometries, dtype=torch.float64)
    if g.ndim == 3:
        g = g[0, ...]
    s = torch.tensor(molecule.perturbed_shieldings, dtype=torch.float64)
    if s.ndim == 3:
        print("Hello!")
        s = s[..., 0]
    if s.ndim == 2:
        s = s[0, ...]
    dn2 = dh.DataNeighbors(x=features, Rs_in=Rs_in, pos=g, r_max=max_radius,
                           self_interaction=True, name=molecule.name, weights=weights, y=s, Rs_out=Rs_out)
    compare_data_neighbors(dn1, dn2)
