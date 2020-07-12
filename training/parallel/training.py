print("loading standard modules...")
import time
from multiprocessing import Process, Queue
from glob import glob
print("loading NN libraries...")
import torch
torch.set_default_dtype(torch.float64)
import torch_geometric as tg
print("loading e3nn...")
import e3nn
import e3nn.point.data_helpers as dh
from e3nn.point.message_passing import Convolution
print("loading training-specific libraries...")
from training_utils import DatasetSignal, DatasetReader, MoleculeProcessor, Molecule
print("done loading modules.")

### configuration ###

# where the raw data are stored
hdf5_filenames = list(sorted(glob("../qm7/data/qm7_*.hdf5")))

# number of examples for test-train split
# the examples picked are strictly in the order they appear
# in the hdf5, but iteration over the set might vary due
# to concurrent processing
testing_size = 100
training_size = 500

# number of concurrent processes that create DataNeighbors
n_molecule_processors = 1

# maximum number of pending molecules
molecule_queue_max_size = 1000

# maximum number of pending DataNeighbors
data_neighbors_queue_max_size = 200

# model parameters

# training parameters
n_epochs = 2               # number of epochs
batch_size = 50            # minibatch sizes
checkpoint_interval = 10   # save model every checkpoint_interval minibatches
learning_rate = 3e-3       # learning rate
max_radius = 5.0           # consider neighbors out to this radius
n_norm = 14.0              # average number of convolution neighbors per atom

### initialize GPU ###
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print(torch.version.cuda)
#print(torch.cuda.memory_summary())
device = "cpu" # "cuda"
temp_tensor = torch.rand(10).to(device)
print(temp_tensor)

### prepare for training ###

# setup processes that will read the training data
# when all examples have been exhausted, n_molecule_processors
# copies of DatasetSignal.STOP will be propagated through molecule_queue
# and data_neighbors_queue to the training loop to signal an end to the epoch
molecule_queue = Queue(molecule_queue_max_size)
data_neighbors_queue = Queue(data_neighbors_queue_max_size)

dataset_reader = DatasetReader("DatasetReader 0", molecule_queue, hdf5_filenames,
                               n_molecule_processors, testing_size)
dataset_reader.start()

# read in and process testing data directly to memory
molecule_processors = [ MoleculeProcessor(f"MoleculeProcessor {i}", molecule_queue,
                        data_neighbors_queue, max_radius) for i in range(n_molecule_processors) ]
for molecule_processor in molecule_processors:
    molecule_processor.start()

testing_data_list = []
stop_signals_received = 0
while True:
    if stop_signals_received == n_molecule_processors and data_neighbors_queue.empty():
       assert len(testing_data_list) == testing_size, \
              "expected {testing_size} testing examples but got {len(testing_data_list)}"
       break

    data_neighbors = data_neighbors_queue.get()
    if data_neighbors == DatasetSignal.STOP:
        stop_signals_received += 1
        continue
    testing_data_list.append(data_neighbors)

testing_dataloader = tg.data.DataListLoader(testing_data_list, batch_size=batch_size, shuffle=False)

# restart molecule processors
dataset_reader.join()           # ensure first pass is finished
datset_reader.reset_counters(training_size)
for molecule_processor in molecule_processors:
    molecule_processor.join()   # ensure first pass is finished
    molecule_processor.start()

### model and optimizer ###

# create model

# create optimizer

### training ###

for epoch in range(epochs):

    # iterate through all training examples
    stop_signals_received = 0
    minibatches_processed = 0
    training_data_list = []
    while True:
        # determine whether all training examples have been seen and trained on
        if stop_signals_received == n_molecule_processors and
           data_neighbors_queue.empty() and len(data_list) == 0:
           break

        # get the next training example
        data_neighbors = data_neighbors_queue.get()
        if data_neighbors == DatasetSignal.STOP:
            stop_signals_received += 1
            continue
        training_data_list.append(data_neighbors)

        # if we have enough for a minibatch, train
        if len(training_data_list) == batch_size:
            data = tg.data.Batch.from_data_list(training_data_list)
            training_data_list = []
            minibatches_processed += 1
            print(f"epoch {epoch+1} minibatch {minibatches_processed}")


