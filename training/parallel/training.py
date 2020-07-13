print("loading standard modules...")
import time
from multiprocessing import Process, Manager
from glob import glob
print("loading torch...")
import torch
torch.set_default_dtype(torch.float64)
print("loading torch_geometric...")
import torch_geometric as tg
print("loading e3nn...")
import e3nn
import e3nn.point.data_helpers as dh
from e3nn.point.message_passing import Convolution
print("loading training-specific libraries...")
import training_config
from training_utils import DatasetSignal, DatasetReader, MoleculeProcessor, Molecule
print("done loading modules.")

### read configuration values ###

all_elements = training_config.all_elements
n_elements = len(all_elements)
relevant_elements = training_config.relevant_elements
hdf5_filenames = training_config.hdf5_filenames
testing_size = training_config.testing_size
training_size = training_config.training_size
n_molecule_processors = training_config.n_molecule_processors
molecule_queue_max_size = training_config.molecule_queue_max_size
data_neighbors_queue_max_size = training_config.data_neighbors_queue_max_size
Rs_in = training_config.Rs_in
Rs_out = training_config.Rs_out
n_epochs = training_config.n_epochs
batch_size = training_config.batch_size
checkpoint_interval = training_config.checkpoint_interval
learning_rate = training_config.learning_rate
max_radius = training_config.max_radius
n_norm = training_config.n_norm

### initialize GPU ###
print(f"\ncurrent cuda device: {torch.cuda.current_device()}")
print(f"cuda device count:   {torch.cuda.device_count()}")
print(f"cuda device name:    {torch.cuda.get_device_name(0)}")
print(f"is cuda available?   {torch.cuda.is_available()}")
print(f"cuda version:        {torch.version.cuda}")
#print(torch.cuda.memory_summary())
device = "cuda"
print(f"device:              {device}")
temp_tensor = torch.rand(10).to(device)
print("test tensor:")
print(temp_tensor)
print()

### prepare for training ###

# setup processes that will read the training data
# when all examples have been exhausted, n_molecule_processors
# copies of DatasetSignal.STOP will be propagated through molecule_queue
# and data_neighbors_queue to the training loop to signal an end to the epoch
manager = Manager()
molecule_queue = manager.Queue(molecule_queue_max_size)
data_neighbors_queue = manager.Queue(data_neighbors_queue_max_size)
dataset_reader_status_queue = manager.Queue()

dataset_reader0 = DatasetReader("DatasetReader 0", molecule_queue, hdf5_filenames,
                                n_molecule_processors, testing_size, dataset_reader_status_queue)
dataset_reader0.start()

# read in and process testing data directly to memory
molecule_processors0 = [ MoleculeProcessor(f"MoleculeProcessor0 {i}", molecule_queue, \
                         data_neighbors_queue, max_radius, Rs_in, Rs_out) \
                         for i in range(n_molecule_processors) ]
for molecule_processor in molecule_processors0:
    molecule_processor.start()

testing_data_list = []
stop_signals_received = 0
while True:
    if stop_signals_received == n_molecule_processors and data_neighbors_queue.empty():
       assert len(testing_data_list) == testing_size, \
              f"expected {testing_size} testing examples but got {len(testing_data_list)}"
       break

    data_neighbors = data_neighbors_queue.get()
    if data_neighbors == DatasetSignal.STOP:
        stop_signals_received += 1
        print("training got a stop, total stops now", stop_signals_received)
        continue
    testing_data_list.append(data_neighbors)

print("final")
for i in testing_data_list:
    print(i)
testing_dataloader = tg.data.DataListLoader(testing_data_list, batch_size=batch_size, shuffle=False)
print("made testing data_loader")
print("-------------------------------------")

### model and optimizer ###

# create model

# create optimizer


### training ###

# start processes that will process training data
dataset_reader1 = DatasetReader("DatasetReader 1", molecule_queue, hdf5_filenames,
                               n_molecule_processors, training_size, None)
state = dataset_reader_status_queue.get()
print(">>> state:", state)
dataset_reader1.hdf5_file_list_index = state[0]
dataset_reader1.hdf5_file_index = state[1]
print(dataset_reader1.hdf5_file_list_index, dataset_reader1.hdf5_file_index)
dataset_reader1.start()

molecule_processors1 = [ MoleculeProcessor(f"MoleculeProcessor1 {i}", molecule_queue, \
                         data_neighbors_queue, max_radius, Rs_in, Rs_out) \
                         for i in range(n_molecule_processors) ]
for molecule_processor in molecule_processors1:
    molecule_processor.start()



# the actual training
for epoch in range(n_epochs):
    print(f"this is epoch {epoch+1}")

    # iterate through all training examples
    stop_signals_received = 0
    minibatches_processed = 0
    training_data_list = []
    while True:
        # determine whether all training examples have been seen and trained on
        last_batch = False
        if stop_signals_received == n_molecule_processors and data_neighbors_queue.empty():
            # if there are no more examples, stop
            if len(training_data_list) == 0:
                break
            last_batch = True
        else:
            # get the next training example
            print(f"stop_signals_received: {stop_signals_received}   queue empty: {data_neighbors_queue.empty()}")
            data_neighbors = data_neighbors_queue.get()
            if data_neighbors == DatasetSignal.STOP:
                stop_signals_received += 1
                print(f"training loop got a stop, stop_signals_recieved is now {stop_signals_received}")
                continue
            print()
            print(data_neighbors)
            print()
            training_data_list.append(data_neighbors)

        # if we have enough for a minibatch, train
        if len(training_data_list) == batch_size or last_batch:
            print("got enough for a batch")
            if last_batch:
                print("last batch")
            data = tg.data.Batch.from_data_list(training_data_list)
            print(f"there are {len(training_data_list)} examples in this batch")
            training_data_list = []
            minibatches_processed += 1
            print(f"epoch {epoch+1} minibatch {minibatches_processed}")

exit()
