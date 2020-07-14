if __name__ != '__main__':
    print("spawning process...")
if __name__ == '__main__': print("loading standard modules...")
import time
from multiprocessing import Manager, Pool, freeze_support
from glob import glob
if __name__ == '__main__': print("loading torch...")
import torch
torch.set_default_dtype(torch.float64)
if __name__ == '__main__': print("loading torch_geometric...")
import torch_geometric as tg
if __name__ == '__main__': print("loading e3nn...")
import e3nn
import e3nn.point.data_helpers as dh
from e3nn.point.message_passing import Convolution
if __name__ == '__main__': print("loading training-specific libraries...")
import training_config
from training_utils import DatasetSignal, DatasetReader, process_molecule, Molecule
if __name__ == '__main__': print("done loading modules.")

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

if __name__ == '__main__': 
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

def main():
    ### prepare for training ###

    # setup processes that will read the training data
    # when all examples have been exhausted, n_molecule_processors
    # copies of DatasetSignal.STOP will be propagated through molecule_queue
    # and data_neighbors_queue to the training loop to signal an end to the epoch
    manager = Manager()
    example_queue = manager.Queue()  # request number of examples to process here, or call for RESTART
    molecule_queue = manager.Queue(molecule_queue_max_size)
    data_neighbors_queue = manager.Queue(data_neighbors_queue_max_size)

    dataset_reader = DatasetReader("dataset_reader", example_queue, molecule_queue,
                                hdf5_filenames, n_molecule_processors)
    example_queue.put((testing_size,True))  # (how many examples to process, whether to make Molecules)
    dataset_reader.start()

    molecule_processor_pool = Pool(n_molecule_processors, process_molecule,
                                (molecule_queue, data_neighbors_queue, max_radius, Rs_in, Rs_out))

    # read in and process testing data directly to memory
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

    # the actual training
    for epoch in range(n_epochs):
        print(f"== this is epoch {epoch+1} ===")

        # reset the counters for reading the input files
        example_queue.put(DatasetSignal.RESTART)

        # skip over the first testing_size examples
        example_queue.put((testing_size, False))

        # process the next training_size examples
        example_queue.put((training_size, True))

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
                print("last batch")
            else:
                # get the next training example
                print(f"stop_signals_received: {stop_signals_received}   queue empty: {data_neighbors_queue.empty()}")
                data_neighbors = data_neighbors_queue.get()
                if data_neighbors == DatasetSignal.STOP:
                    stop_signals_received += 1
                    print(f"training loop got a stop, stop_signals_received is now {stop_signals_received}")
                else:
                    print()
                    print(data_neighbors)
                    print()
                    training_data_list.append(data_neighbors)

            # if we have enough for a minibatch, train
            if len(training_data_list) == batch_size or last_batch:
                print("got enough for a batch")
                time1 = time.time()
                data = tg.data.Batch.from_data_list(training_data_list)
                time2 = time.time()
                elapsed = time2-time1
                print(f"batch took {elapsed:.3f} s to make")
                print(f"there are {len(training_data_list)} examples in this batch")
                training_data_list = []
                minibatches_processed += 1
                print(f"epoch {epoch+1} minibatch {minibatches_processed} is finished")

            # stop if this is the last batch
            if last_batch:
                break

    # clean up
    example_queue.put(DatasetSignal.STOP)
    molecule_processor_pool.close()
    molecule_processor_pool.terminate()
    molecule_processor_pool.join()
    print("all done")


if __name__ == '__main__':
    freeze_support()
    main()


