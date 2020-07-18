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

### Functions for Training ###

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

# saves a model and optimizer to disk
def checkpoint(model_kwargs, model, filename, optimizer):
    model_dict = {
        'state_dict' : model.state_dict(),
        'model_kwargs' : model_kwargs,
        'optimizer_state_dict' : optimizer.state_dict()
    }
    printf("Checkpointing to {filename}...", end='', flush=True)
    torch.save(model_dict, filename)
    file_size = os.path.getsize(filename) / 1E6
    printf("occupies {file_size:.2f} MB.")

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

class Pipeline():
    def __init__(self, hdf5_filenames, requested_jiggles, n_molecule_processors,
                 max_radius, Rs_in, Rs_out, max_size=None, manager=None):
        if manager is None:
            manager = Manager()
        #self.manager = manager
        self.command_queue = manager.Queue(max_size)
        self.molecule_queue = manager.Queue(max_size)
        self.data_neighbors_queue = manager.Queue(max_size)
        self.molecule_processor_pool = Pool(n_molecule_processors, process_molecule,
                                            (self, max_radius, Rs_in, Rs_out))
        self.testing_molecules_dict = manager.dict()
        self.dataset_reader = DatasetReader("dataset_reader", self, hdf5_filenames,
                                            requested_jiggles, self.testing_molecules_dict)

        self.knows = Semaphore(0) # > 0 if we know if any are coming
        self.finished_reading = Lock() # locked if we're still reading from file
        self.in_pipe = Value('i',0) # number of molecules that have been sent to the pool

        self.dataset_reader.start()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        self_dict['molecule_processor_pool'] = None
        return self_dict

    # methods for pipeline user/consumer:
    def start_reading(self, examples_to_read, make_molecules, record_in_dict):
        #print("Start reading...")
        assert check_semaphore(self.finished_reading), "Tried to start reading file, but already reading!"
        with self.in_pipe.get_lock():
            assert self.in_pipe.value == 0, "Tried to start reading, but examples already in pipe!"
        set_semaphore(self.finished_reading, False)
        set_semaphore(self.knows, False)
        self.command_queue.put((examples_to_read, make_molecules, record_in_dict))

    def wait_till_done(self):
        wait_semaphore(self.knows)
        wait_semaphore(self.finished_reading)
        assert not self.any_coming(), "Pipeline consumer is waiting on pipeline, but not getting from pipe!"

    def restart(self):
        assert not self.any_coming() , "Tried to restart pipeline before it was empty!"
        self.command_queue.put(DatasetSignal.RESTART)
        # What to do if things are still in the pipe???

    def any_coming(self): # returns True if at least one example is coming
        wait_semaphore(self.knows)
        with self.in_pipe.get_lock():
            return self.in_pipe.value > 0

    def get_data_neighbor(self, timeout=None):
        assert self.any_coming(), "Tried to get an example from an empty pipeline!"
        x = self.data_neighbors_queue.get(True, timeout)
        with self.in_pipe.get_lock():
            self.in_pipe.value -= 1
            if self.in_pipe.value == 0 and not check_semaphore(self.finished_reading):
                set_semaphore(self.knows, False)
            return x

    def close(self):
        self.command_queue.put(DatasetSignal.STOP)
        self.molecule_processor_pool.close()
        time.sleep(2)
        self.molecule_processor_pool.terminate()
        self.molecule_processor_pool.join()

    # methods for DatasetReader:
    def get_command(self):
        return self.command_queue.get()

    def put_molecule(self, m):
        self.molecule_queue.put(m)
        with self.in_pipe.get_lock():
            self.in_pipe.value += m.perturbed_geometries.shape[0]
            set_semaphore(self.knows, True)

    def set_finished_reading(self): # !!! Call only after you've put the molecules !!!
        set_semaphore(self.finished_reading, True)
        set_semaphore(self.knows, True)

    # methods for molecule processors:
    def get_molecule(self):
        return self.molecule_queue.get()

    def put_data_neighbor(self, x):
        self.data_neighbors_queue.put(x)

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
    def __init__(self, name, pipeline,
                       hdf5_filenames,
                       requested_jiggles,
                       testing_molecules_dict):
        super().__init__(group=None, target=None, name=name)
        self.pipeline = pipeline
        self.hdf5_filenames = hdf5_filenames   # hdf5 files to process
        self.hdf5_file_list_index = 0          # which hdf5 file
        self.hdf5_file_index = 0               # which example within the hdf5 file
        self.requested_jiggles = requested_jiggles  # how many jiggles per file
        self.testing_molecules_dict = testing_molecules_dict   # molecule name -> Molecule

    # process the data in all hdf5 files
    def run(self):
        assert len(self.hdf5_filenames) > 0, "no files to process!"

        while True:
            command = self.pipeline.get_command()
            #print(f"Command: {command}")
            if command == DatasetSignal.RESTART:
                self.hdf5_file_list_index = 0
                self.hdf5_file_index = 0
                continue
            elif command == DatasetSignal.STOP:
                break
            elif isinstance(command, tuple):
                assert len(command) == 3, \
                       "expected 3-tuple: (examples_to_process, make_molecules, record_in_dict)"
                examples_to_process, make_molecules, record_in_dict = command
                self.read_examples(examples_to_process, make_molecules, self.requested_jiggles, record_in_dict)
                self.pipeline.set_finished_reading()
            else:
                raise ValueError("unexpected work type")

    # iterate through hdf5 filenames, picking up where we left off
    # returns: number of examples processed
    def read_examples(self, examples_to_read, make_molecules, requested_jiggles, record_in_dict):
        examples_processed = 0       # how many examples have been processed this round
        assert self.hdf5_file_list_index < len(self.hdf5_filenames), \
            "request to read examples, but files are finished!"
        while examples_processed < examples_to_read:
            hdf5_filename = self.hdf5_filenames[self.hdf5_file_list_index]
            print(f"{self.name}: filename={hdf5_filename} file_list_index={self.hdf5_file_list_index} file_index={self.hdf5_file_index}")
            examples_processed += self.read_hdf5(hdf5_filename, examples_to_read - examples_processed, make_molecules, requested_jiggles, record_in_dict)
            if self.hdf5_file_list_index >= len(self.hdf5_filenames):
                break
        return examples_processed

    # process the data in this hdf5 file
    # requested_jiggles examples will be taken: 1 (default) or listlike
    # make_molecules: boolean that tells us if we should make Molecules objects
    #                 or just skip over these records
    # returns: number of molecules read from this file
    def read_hdf5(self, filename, examples_to_read, make_molecules, requested_jiggles, record_in_dict):
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
                    jiggles = list(range(requested_jiggles))
                elif isinstance(requested_jiggles, list):
                    jiggles = requested_jiggles

                jiggles_needed = examples_to_read - examples_read
                assert jiggles_needed >= 1
                if jiggles_needed < len(jiggles):
                    jiggles = jiggles[:jiggles_needed]

                # NOTE: We don't split a single molecule between different batches!
                perturbed_geometries = geometries_and_shieldings[jiggles,:,:3]
                perturbed_shieldings = geometries_and_shieldings[jiggles,:,3]
                n_examples, n_atoms, _ = np.shape(perturbed_geometries)

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
                    self.pipeline.put_molecule(molecule)
                    if record_in_dict:
                        self.testing_molecules_dict[molecule.name] = molecule
                    #print(f"put molecule {molecule.name}, n_examples={n_examples}")

                # update counters
                examples_read += n_examples
                #print("examples_read:", examples_read)
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
def process_molecule(pipeline, max_radius, Rs_in, Rs_out):
    while True:
        molecule = pipeline.get_molecule()
        #print(f"> got molecule {molecule.name}, n_examples={len(molecule.perturbed_geometries)}")
        assert isinstance(molecule, Molecule), \
               f"expected Molecule but got {type(work)} instead!"

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

### Training Code ###

class TrainingHistory():
    def __init__(self):
        self.start_time = time.time()
        self.minibatch_epochs = []   # which epoch each minibatch was computed in
        self.minibatch_times = []    # times in seconds since start of training
        self.minibatches_seen = []   # how many minibatches have been seen this epoch
        self.minibatch_losses = []
        self.testing_epochs = []     # which epoch the testing was computed in
        self.testing_times = []      # times in seconds since start of training
        self.testing_losses = []

    def log_minibatch_loss(self, epoch, minibatches_seen, minibatch_loss):
        elapsed_time = time.time() - self.start_time
        self.minibatch_epochs.append(epoch)
        self.minibatch_times.append(elapsed_time)
        self.minibatches_seen.append(minibatches_seen)
        self.minibatch_losses.append(minibatch_loss)

    def log_testing_loss(epoch, testing_loss):
        elapsed_time = time.time() - self.start_time
        self.testing_epochs.append(epoch)

# train a single batch
def train_batch(data_list, n_minibatches, model, optimizer, training_history, epoch, minibatches_seen):
    # forward pass
    time1 = time.time()
    data = tg.data.Batch.from_data_list(data_list)
    data.to(device)
    time2 = time.time()
    output = model(data.x, data.edge_index, data.edge_attr, n_norm=n_norm)
    loss, _ = loss_function(output,data)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update results
    minibatch_loss = np.sqrt(loss.item())  # RMSE
    training_history.log_minibatch_loss(epoch, minibatches_seen, minibatch_loss)
    recent_minibatch_losses = np.array(training_history.minibatch_losses[-10:])
    training_moving_average_loss = np.mean(recent_minibatch_losses)
    time3 = time.time()
    print(f"Epoch {epoch:<4d}    train {minibatches_seen+1:5d} / {n_minibatches:5d}  loss = {training_moving_average_loss:12.3f}  batchcopytime = {time2-time1:.2f} s   train_time = {time3-time2:.2f}               ", end="\r", flush=True)

# compute the testing losses
# testing_molecule_dict: name -> Molecule
# returns: testing_loss,
#          results_dict (molecule name -> residuals (n_examples,n_atoms),
#          results_dict2 (site label -> residuals)
#          RMSE_dict (element -> RMSE)
def compute_testing_loss(testing_dataloader, training_history, epoch, molecule_dict):
    n_minibatches = math.ceil(testing_size/batch_size)
    testing_loss = 0.0
    n_testing_eaxmples_seen = 0
    for minibatch_index, data_list in enumerate(testing_dataloader):
        n_examples_this_minibatch = len(data)
        data = tg.data.Batch.from_data_list(data_list)
        data.to(device)

        with torch.no_grad():
            # run model
            output = model(data.x, data.edge_index, data.edge_attr)

            # compute MSE
            loss, residuals = loss_function(output_data)
            minibatch_loss = np.sqrt(loss.item())
            testing_loss = testing_loss * n_testing_examples_seen + \
                           minibatch_loss * n_examples_this_minibatch
            n_testing_examples_seen += n_examples_this_minibatch
            testing_loss = testing_loss / n_testing_examples_seen

            # store residuals

        print(f"Epoch {epoch+1:<4d}    test {minibatch_index+1:5d} / {n_minibatches:5d}  loss = {testing_loss:12.3f}                    ", end="\r", flush=True)

    # reshape residual data

    training_history.log_testing_loss(epoch, testing_loss)
