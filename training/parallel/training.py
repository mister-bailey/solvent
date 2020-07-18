import training_config
if __name__ != '__main__':
    print("spawning process...")
if __name__ == '__main__': print("loading standard modules...")
import time
from glob import glob
import os
import math
if __name__ == '__main__': print("loading torch...")
import torch
from torch.multiprocessing import freeze_support
torch.set_default_dtype(torch.float64)
if __name__ == '__main__': print("loading torch_geometric...")
import torch_geometric as tg
if __name__ == '__main__': print("loading e3nn...")
import e3nn
import e3nn.point.data_helpers as dh
from e3nn.point.message_passing import Convolution
if __name__ == '__main__': print("loading training-specific libraries...")
from training_utils import Pipeline, Molecule, TrainingHistory, train_batch, compute_testing_loss
from diagnostics import print_parameter_size, count_parameters, get_object_size
from variable_networks import VariableParityNetwork
if __name__ == '__main__': print("done loading modules.")

if os.name == 'posix':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"\nMaximum # of open file descriptors: {rlimit[0]} (soft limit) / {rlimit[1]} (hard limit)")

### read configuration values ###

device = training_config.device
all_elements = training_config.all_elements
n_elements = len(all_elements)
relevant_elements = training_config.relevant_elements
hdf5_filenames = training_config.hdf5_filenames
jiggles_per_molecule = training_config.jiggles_per_molecule
testing_size = training_config.testing_size
training_size = training_config.training_size
n_molecule_processors = training_config.n_molecule_processors
molecule_queue_max_size = training_config.molecule_queue_max_size
data_neighbors_queue_max_size = training_config.data_neighbors_queue_max_size
Rs_in = training_config.Rs_in
Rs_out = training_config.Rs_out
n_epochs = training_config.n_epochs
batch_size = training_config.batch_size
testing_interval = training_config.testing_interval
checkpoint_interval = training_config.checkpoint_interval
checkpoint_prefix = training_config.checkpoint_prefix
load_model_from_file = training_config.load_model_from_file
n_norm = training_config.n_norm

if __name__ == '__main__':
    ### initialize GPU ###
    print("\n=== GPU settings: ===\n")
    print(f"current cuda device: {torch.cuda.current_device()}")
    print(f"cuda device count:   {torch.cuda.device_count()}")
    print(f"cuda device name:    {torch.cuda.get_device_name(0)}")
    print(f"is cuda available?   {torch.cuda.is_available()}")
    print(f"cuda version:        {torch.version.cuda}")
    print(f"device:              {device}")
    #print(torch.cuda.memory_summary())
    temp_tensor = torch.rand(10).to(device)
    print("test tensor:")
    print(temp_tensor)
    print()

def main():
    ### initialization ###

    # load pre-existing model if requested
    if load_model_from_file == False:
        muls = training_config.muls
        lmaxes = training_config.lmaxes
        max_radius = training_config.max_radius
        learning_rate = training_config.learning_rate
        number_of_basis = training_config.number_of_basis
        model = None
        optimizer = None
    else:
        model_filename = load_model_from_file
        if not os.path.exists(model_filename):
            print(f"Could not find serialized model '{model_filename}'!")
            exit()
        print(f"Loading model from {model_filename}...")
        model_dict = torch.load(model_filename)
        model_kwargs = model_dict['model_kwargs']
        muls = model_kwargs['muls']
        lmaxes = model_kwargs['lmaxes']
        model = VariableParityNetwork(convolution=Convolution, **model_kwargs)
        model.load_state_dict(model_dict['state_dict'])
        learning_rate = model_dict['optimizer_state_dict']['lr']
        optimizer = torch.optim.Adam(model.parameters(), 0.1)
        optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        number_of_basis = model_kwargs['number_of_basis']

    # report configuration
    print("\n=== Configuration ===\n")
    print("all_elements:                     ", all_elements)
    print("relevant_elements:                ", relevant_elements)
    print("jiggles_per_molecule:             ", jiggles_per_molecule)
    print("testing_size:                     ", testing_size)
    print("training_size:                    ", training_size)
    print("n_molecule_processors:            ", n_molecule_processors)
    print("molecule_queue_max_size:          ", molecule_queue_max_size)
    print("data_neighbors_queue_max_size:    ", data_neighbors_queue_max_size)
    print("Rs_in:                            ", Rs_in)
    print("Rs_out:                           ", Rs_out)
    print("n_epochs:                         ", n_epochs)
    print("batch_size:                       ", batch_size)
    print("testing_interval:                 ", testing_interval)
    print("checkpoint_interval:              ", checkpoint_interval)
    print("checkpoint_prefix:                ", checkpoint_prefix)
    print("learning_rate:                    ", learning_rate)
    print("muls:                             ", muls)
    print("lmaxes:                           ", lmaxes)
    print("max_radius:                       ", max_radius)
    print("number_of_basis:                  ", number_of_basis)
    print("n_norm:                           ", n_norm)
    print()

    print(f"Will use training data from {len(hdf5_filenames)} files:")
    for filename in hdf5_filenames:
        print(f"   {filename}")

    ### prepare for training ###

    # setup processes that will read the training data
    # when all examples have been exhausted, n_molecule_processors
    # copies of DatasetSignal.STOP will be propagated through molecule_queue
    # and data_neighbors_queue to the training loop to signal an end to the epoch
    print("\n=== Preprocessing Testing Data ===\n")
    time1 = time.time()
    pipeline = Pipeline(hdf5_filenames, jiggles_per_molecule, n_molecule_processors,
                        max_radius, Rs_in, Rs_out, molecule_queue_max_size)
    testing_molecules_dict = pipeline.testing_molecules_dict
    pipeline.start_reading(testing_size,True,True)  # (how many examples to process,
                                                    #  whether to make Molecules,
                                                    #  whether to save the molecules to a dict)

    # read in and process testing data directly to memory
    testing_data_list = []

    while pipeline.any_coming():
        data_neighbor = pipeline.get_data_neighbor()
        testing_data_list.append(data_neighbor)
    assert len(testing_data_list) == testing_size, \
        f">>>>> expected {testing_size} testing examples but got {len(testing_data_list)}"

    testing_dataloader = tg.data.DataListLoader(testing_data_list, batch_size=batch_size, shuffle=False)
    time2 = time.time()
    print(f"Done preprocessing testing data!  That took {time2-time1:.3f} s.\n")
    testing_molecules_dict = dict(testing_molecules_dict)

    ### model and optimizer ###

    # create model if it wasn't loaded from disk
    if model == None:
        model_kwargs = {
            'Rs_in' : Rs_in,
            'Rs_out' : Rs_out,
            'muls' : muls,
            'lmaxes' : lmaxes,
            'max_radius' : max_radius,
            'number_of_basis' : number_of_basis,
        }
        model = VariableParityNetwork(convolution=Convolution, batch_norm=True, **model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # print model details
    print("=== Model and Optimizer ===\n")
    model_size = get_object_size(model) / 1E6
    optimizer_size = get_object_size(optimizer) / 1E6
    print(f"Model occupies {model_size:.2f} MB and optimizer occupies {optimizer_size:.2f} MB.\n")
    print("Model Details:")
    print_parameter_size(model)
    print()
    print("Parameters per Layer:")
    count_parameters(model)
    print()
    print("Optimizer:")
    print(optimizer)
    model.to(device)

    ### training ###

    # the actual training
    training_start_time = time.time()
    for epoch in range(1,n_epochs+1):
        print(f"=== Epoch {epoch} ===")

        # reset the counters for reading the input files
        pipeline.restart()

        # skip over the first testing_size examples
        pipeline.start_reading(testing_size, False, False)
        pipeline.wait_till_done()

        # process the next training_size examples
        pipeline.start_reading(training_size, True, False)

        # iterate through all training examples
        minibatches_seen = 0
        checkpoint_index = 0
        training_data_list = []
        n_minibatches = math.ceil(training_size/batch_size)
        training_history = TrainingHistory()
        while pipeline.any_coming():
            time1 = time.time()
            while pipeline.any_coming() and len(training_data_list) < batch_size:
                data_neighbors = pipeline.get_data_neighbor()
                training_data_list.append(data_neighbors)
            wait_time = time.time()-time1

            # determine whether all training examples have been seen and trained on
            if len(training_data_list) > 0:
                data = tg.data.Batch.from_data_list(training_data_list)
                train_batch(training_data_list, n_minibatches, model, optimizer,
                            training_history, epoch, minibatches_seen)
                print(f"\nwait time: {wait_time:.2f}  molecule_queue {pipeline.molecule_queue.qsize()}  data_neighbors_queue {pipeline.data_neighbors_queue.qsize()}")
                training_data_list = []
                minibatches_seen += 1

                #if minibatches_seen % testing_interval:
                #    compute_testing_loss(testing_dataloader, training_history,
                #                         epoch, testing_molecule_dict)

                #if minibatches_seen % checkpoint_interval:
                #    checkpoint_filename = f"{checkpoint_prefix}-epoch_{epoch}-chk_{checkpoint_index}.torch"
                #    checkpoint(model_kwargs, model, checkpoint_filename, optimizer)
                #    checkpoint_index += 1

    # clean up
    training_stop_time = time.time()
    training_elapsed_time = training_stop_time - training_start_time
    print(f"Training took {training_elapsed_time:.3f} s.")
    pipeline.close()

    print("all done")

if __name__ == '__main__':
    freeze_support()
    main()
