if __name__ != '__main__':
    print("spawning process...")
if __name__ == '__main__': print("loading standard modules...")
import time
from multiprocessing import Manager, Pool, freeze_support, Lock #Value
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
from training_utils import DatasetSignal, DatasetReader, process_molecule, Molecule, PipelineReporter
if __name__ == '__main__': print("done loading modules.")


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print(rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

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

    pipeline_reporter = PipelineReporter()#(manager)
    dataset_reader = DatasetReader("dataset_reader", example_queue, molecule_queue,
                                pipeline_reporter, hdf5_filenames)
    example_queue.put((testing_size,True))  # (how many examples to process, whether to make Molecules)
    dataset_reader.start()

    molecule_processor_pool = Pool(n_molecule_processors, process_molecule,
                                (molecule_queue, data_neighbors_queue, max_radius, Rs_in, Rs_out))

    # read in and process testing data directly to memory
    testing_data_list = []

    while pipeline_reporter.any_coming():
        data_neighbors = data_neighbors_queue.get()
        pipeline_reporter.take_from_pipe()
        testing_data_list.append(data_neighbors)
    assert len(testing_data_list) == testing_size, \
        f"expected {testing_size} testing examples but got {len(testing_data_list)}"

    #print("final")
    #for i in testing_data_list:
    #    print(i)
    testing_dataloader = tg.data.DataListLoader(testing_data_list, batch_size=batch_size, shuffle=False)
    #print("made testing data_loader")
    #print("-------------------------------------")

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
        minibatches_processed = 0
        training_data_list = []
        while pipeline_reporter.any_coming():
            while pipeline_reporter.any_coming() and len(training_data_list) < batch_size:
                data_neighbors = data_neighbors_queue.get()
                pipeline_reporter.take_from_pipe()
                training_data_list.append(data_neighbors)

            # determine whether all training examples have been seen and trained on
            if len(training_data_list) > 0:
                #print("got enough for a batch")
                time1 = time.time()
                data = tg.data.Batch.from_data_list(training_data_list)
                time2 = time.time()
                elapsed = time2-time1
                print(f"batch took {elapsed:.3f} s to make")
                #print(f"there are {len(training_data_list)} examples in this batch")
                training_data_list = []
                minibatches_processed += 1
                #print(f"epoch {epoch+1} minibatch {minibatches_processed} is finished")

    # clean up
    example_queue.put(DatasetSignal.STOP)
    molecule_processor_pool.close()
    time.sleep(2)
    molecule_processor_pool.terminate()
    molecule_processor_pool.join()
    print("all done")

if __name__ == '__main__':
    freeze_support()
    main()
