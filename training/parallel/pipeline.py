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

class Molecule():
    def __init__(self, ID, smiles,
                 perturbed_geometries,
                 perturbed_shieldings,
                 atomic_numbers,
                 symmetrical_atoms=None,        # list of lists of 0-indexed atom numbers
                 weights=None):
        self.ID = ID     # database id for molecule
        self.smiles = smiles
        # vector of strings of length n_atoms
        self.atomic_numbers = atomic_numbers
        # number of atoms
        self.n_atoms = len(atomic_numbers)
        # (n_examples, n_atoms, 3)
        self.perturbed_geometries = perturbed_geometries
        self.perturbed_shieldings = perturbed_shieldings
        #print(self.perturbed_shieldings.shape)   

        self.features = Molecule.get_one_hots(atomic_numbers)
        if weights is None:
            self.weights = Molecule.get_weights(
                atomic_numbers, symmetrical_atoms, None)   # (n_atoms,)
        else:
            self.weights = weights

    one_hot_table = np.zeros((0,0))

    # initialize one-hot table
    # also initalizes revers look-up for atomic numbers
    @staticmethod
    def initialize_one_hot_table(all_elements):
        max_element = max(all_elements)
        Molecule.one_hot_table = np.zeros((max_element+1, len(all_elements)), dtype=np.float64)
        Molecule.atomic_number_index = np.zeros(len(all_elements), dtype=np.int32)
        for i, e in enumerate(all_elements):
            Molecule.one_hot_table[e][i] = 1.0
            Molecule.atomic_number_index[i] = e

    # generates one-hots for a list of atomic_numbers
    @staticmethod
    def get_one_hots(atomic_numbers):
        return Molecule.one_hot_table[atomic_numbers]

    # get atomic number(s) from one-hots
    @staticmethod
    def get_atomic_numbers(one_hots):
        return one_hots @ Molecule.atomic_number_index

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
        if new_process == True and manager is None:
            manager = Manager()
        self.knows = Semaphore(0)  # > 0 if we know if any are coming
        # == 0 if DatasetReader is processing a command
        self.working = Semaphore(1 if new_process else 100)
        self.finished_reading = Lock()  # locked if we're still reading from file
        # number of molecules that have been sent to the pipe:
        self.in_pipe = Value('i', 0)
        
        # Tracking what's already been sent through the pipe:
        self._example_number = Value('i', 0)
        
        # The final kill switch:
        self._close = Value('i', 0)

        self.command_queue = manager.Queue(10)
        self.molecule_pipeline = None
        self.batch_queue = Queue(config.data.batch_queue_cap) #manager.Queue(config.data.batch_queue_cap)
        self.share_batches = share_batches

        self.dataset_reader = DatasetReader("dataset_reader", self, config, new_process=new_process)
        if new_process:
            self.dataset_reader.start()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        self_dict['dataset_reader'] = None
        return self_dict

    # methods for pipeline user/consumer:
    def start_reading(self, examples_to_read, make_molecules=True, batch_size=None, wait=False):
        #print("Start reading...")
        assert check_semaphore(
            self.finished_reading), "Tried to start reading file, but already reading!"
        with self.in_pipe.get_lock():
            assert self.in_pipe.value == 0, "Tried to start reading, but examples already in pipe!"
        set_semaphore(self.finished_reading, False)
        set_semaphore(self.knows, False)
        self.working.acquire()
        self.command_queue.put(StartReading(
            examples_to_read, make_molecules, batch_size))
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

    def scan_to(self, index):
        assert check_semaphore(
            self.knows), "Tried to scan to index, but don't know if finished!"
        assert check_semaphore(
            self.finished_reading), "Tried to scan to index, but not finished reading!"
        assert not self.any_coming(), "Tried to scan to index, but pipeline not empty!"
        self.working.acquire()
        self.command_queue.put(ScanTo(index))
        with self._example_number.get_lock():
            self._example_number.value = index
        # What to do if things are still in the pipe???

    def set_indices(self, test_set_indices):
        self.working.acquire()
        self.command_queue.put(SetIndices(torch.tensor(test_set_indices)))
        self.working.acquire()
        self.command_queue.put(ScanTo(0))

    def set_shuffle(self, shuffle):
        self.command_queue.put(SetShuffle(shuffle))

    def any_coming(self):  # returns True if at least one example is coming
        wait_semaphore(self.knows)
        with self.in_pipe.get_lock():
            return self.in_pipe.value > 0

    def get_batch(self, timeout=None):
        #assert self.any_coming(verbose=verbose), "Tried to get data from an empty pipeline!"
        x = self.batch_queue.get(True, timeout)
        with self.in_pipe.get_lock():
            self.in_pipe.value -= x.n_examples
            if self.in_pipe.value == 0 and not check_semaphore(self.finished_reading):
                set_semaphore(self.knows, False)
        with self._example_number.get_lock():
            self._example_number.value += x.n_examples
        return x

    @property
    def example_number(self):
        with self._example_number.get_lock():
            return self._example_number.value

    def close(self):
        self.command_queue.put(CloseReader())
        with self._close.get_lock():
            self._close.value = True
        self.dataset_reader.join(2)
        self.dataset_reader.kill()

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

    def put_molecule_data(self, data, atomic_numbers, weights, ID, block=True):
        r = self.molecule_pipeline.put_molecule_data(
            data, atomic_numbers, weights, ID, block)
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
        
    def time_to_close(self):
        with self._close.get_lock():
            return self._close.value

class DatasetSignal():
    def __str__(self):
        return "DatasetSignal"


class ScanTo(DatasetSignal):
    def __init__(self, index=0):
        self.index=0

    def __str__(self):
        return f"ScanTo({self.index})"
        
class CloseReader(DatasetSignal):
    def __init__(self, aggressive=False):
        self.aggressive = aggressive


class StartReading(DatasetSignal):
    def __init__(self, examples_to_read, make_molecules=True, batch_size=None):
        self.examples_to_read = examples_to_read
        self.make_molecules = make_molecules
        #self.record_in_dict = record_in_dict
        self.batch_size = batch_size

    def __str__(self):
        r = f"StartReading(examples_to_read={self.examples_to_read}, make_molecules={self.make_molecules}"
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
        
class SetShuffle(DatasetSignal):
    def __init__(self, shuffle=True):
        self.shuffle_incoming = shuffle
    
    def __str__(self):
        return f"SetShuffle({self.shuffle_incoming})"
        

class DatasetReader(Process):
    def __init__(self, name, pipeline, config, shuffle_incoming=False, new_process=True, requested_jiggles=1):
        if new_process:
            super().__init__(group=None, target=None, name=name)
        self.new_process = new_process
        self.pipeline = pipeline
        self.config = config

        self.hdf5_file_list_index = 0          # which hdf5 file
        self.hdf5_file_index = 0               # which example within the hdf5 file
        self.all_elements = config.all_elements
        self.requested_jiggles = requested_jiggles  # how many jiggles per file
        #self.testing_molecules_dict = testing_molecules_dict   # ID -> Molecule
        self.molecule_pipeline = None
        self.use_tensor_constraint = config.training.use_tensor_constraint
        feature_size = len(config.all_elements)
        output_size = 10 if self.use_tensor_constraint else 1
        self.molecule_pipeline_args = (config.training.batch_size, config.max_radius, feature_size,
                                       output_size, config.data.n_molecule_processors,
                                       config.data.molecule_queue_cap, config.data.example_queue_cap,
                                       config.data.batch_queue_cap, config.affine_correction)
        #print(f"molecule_pipeline_args = {self.molecule_pipeline_args}")
        #self.molecule_number = 0
        self.index_pos = 0
        self.shuffle_incoming = shuffle_incoming

        self.data_source = config.data.source
        if self.data_source == 'hdf5':
            self.hdf5_filenames = config.data.hdf5_filenames   # hdf5 files to process
            self.read_examples = self.read_examples_from_file
            if config.data.file_format == 0:
                self.read_hdf5 = self.read_hdf5_format_0
            elif config.data.file_format == 1:
                self.read_hdf5 = self.read_hdf5_format_1
        elif self.data_source == 'SQL':
            self.connect_params = config.data.connect_params
            self.SQL_fetch_size = config.data.SQL_fetch_size
            self.molecule_buffer = []
            self.read_examples = self.read_examples_from_SQL
            from mysql_df import MysqlDB
            self.database = MysqlDB(self.connect_params)

    # command loop for reader:
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
            if isinstance(command, ScanTo):
                # move the reader head to command.index
                if self.data_source == 'hdf5':
                    self.hdf5_file_list_index = 0
                    self.hdf5_file_index = 0
                    # hdf5 can only scan by reading through the files
                    if command.index > 0:
                        self.read_examples(command.index, False, False)
                        self.pipeline.set_finished_reading()
                elif self.data_source == 'SQL':
                    self.index_pos = command.index
                    self.molecule_buffer = []
                #self.molecule_number = 0
            elif isinstance(command, StartReading):
                self.molecule_pipeline.notify_starting(command.batch_size)
                self.read_examples(command.examples_to_read, command.make_molecules)
                self.pipeline.set_finished_reading()
                while self.molecule_pipeline.any_batch_coming():
                    self.pipeline.put_batch(self.pipeline.get_batch_from_ext())
            elif isinstance(command, SetIndices):
                self.indices = command.indices
            elif isinstance(command, SetShuffle):
                self.shuffle_incoming = command.shuffle_incoming
            elif isinstance(command, CloseReader):
                return
            else:
                raise ValueError("unexpected work type")
            self.pipeline.working.release()
            if not self.new_process:
                break

    # iterate through hdf5 filenames, picking up where we left off
    # returns: number of examples processed
    def read_examples_from_file(self, examples_to_read, make_molecules):
        examples_read = 0       # how many examples have been processed this round
        assert self.hdf5_file_list_index < len(self.hdf5_filenames), \
            "request to read examples, but files are finished!"
        while examples_read < examples_to_read:
            hdf5_filename = self.hdf5_filenames[self.hdf5_file_list_index]
            #print(f"{self.name}: filename={hdf5_filename} file_list_index={self.hdf5_file_list_index} file_index={self.hdf5_file_index}")
            examples_read += self.read_hdf5(
                hdf5_filename, examples_to_read - examples_read, make_molecules)
            if self.hdf5_file_list_index >= len(self.hdf5_filenames):
                break
        return examples_read

    # I've removed the original hdf5 reader, since we don't use that format any more
    # you can find it on the github
    def read_hdf5_format_0(self, filename, examples_to_read, make_molecules):
        raise Exception("Old hdf5 format not supported!")

    def read_hdf5_format_1(self, filename, examples_to_read, make_molecules):
        with h5py.File(filename, "r") as h5:
            if make_molecules:
                examples_read = 0
                for key, dataset in itertools.islice(h5.items(), self.hdf5_file_index, None):
                    molecule = Molecule(int(key), str(dataset.attrs["smiles"]), dataset[..., :3],
                            dataset[..., 3], dataset.attrs["atomic_numbers"],
                            weights=dataset.attrs["weights"])
                    self.pipeline.put_molecule_to_ext(molecule)

                    # ABORT if we are exiting training
                    if self.pipeline.time_to_close():
                        return examples_to_read

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

    def read_examples_from_SQL(self, examples_to_read, make_molecules):
        examples_read = 0
        while examples_read < examples_to_read:
            i=0
            for i, (ID, data, weights, smiles) in enumerate(self.molecule_buffer):
                if make_molecules:
                    if examples_read == examples_to_read:
                        break
                    molecule = Molecule(ID, str(smiles), data[:, 1:4], data[:, 4:],
                                        data[:, 0].astype(np.int32), weights=weights)
                    #print(f"# ID: {molecule.ID}")
                    self.pipeline.put_molecule_to_ext(molecule)
                    #if record_in_dict:
                    #    self.testing_molecules_dict[molecule.ID] = molecule
                examples_read += 1
                
                # ABORT if we are exiting training
                if self.pipeline.time_to_close():
                    return examples_to_read
                    
                while self.pipeline.ext_batch_ready():
                    bad_call = self.pipeline.get_batch_from_ext()
                    self.pipeline.put_batch(bad_call)
            self.molecule_buffer = self.molecule_buffer[i:]
            if len(self.molecule_buffer) < self.SQL_fetch_size and self.index_pos < len(self.indices):
                self.molecule_buffer += self.database.read_rows(np.nditer(
                    self.indices[self.index_pos : self.index_pos + self.SQL_fetch_size]),
                    randomize = self.shuffle_incoming, get_tensors=self.use_tensor_constraint)
                self.index_pos += self.SQL_fetch_size

        return examples_read


# returns a random shuffle of the available indices, for test/train split and random training
def generate_index_shuffle(size, connect_params, rng=None, seed=None, randomize=True, get_from_start=False):
    from mysql_df import MysqlDB
    db = MysqlDB(connect_params)
    indices = np.array(db.get_finished_idxs(size, ordered=get_from_start), dtype=np.int32)
    if randomize:
        if rng is None:
            rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    else:
        indices = np.sort(indices)
    return indices

# allow ordering within jiggle group??
def generate_multi_jiggles_set(n_molecules, n_jiggles, connect_params, randomize=True,
                               get_from_start=False, rng=None, seed=None):
    from mysql_df import MysqlDB
    db = MysqlDB(connect_params)

    indices = np.array(db.get_columns_with_cond('id', f'mod(id, 1000) < {n_jiggles}', n_molecules * n_jiggles))
    assert len(indices) == n_molecules * n_jiggles, "Couldn't get requested number of jiggles!"

    if randomize:
        if rng is None:
            rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    else:
        indices.sort()
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
                                  self_interaction=True, ID=molecule.ID,
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
    molecule = molecule_dict[dn1.ID]
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
                           self_interaction=True, ID=molecule.ID, weights=weights, y=s, Rs_out=Rs_out)
    compare_data_neighbors(dn1, dn2)
