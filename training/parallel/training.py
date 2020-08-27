import training_config as config
if __name__ != '__main__':
    print("spawning process...")
if __name__ == '__main__': print("loading standard modules...")
import time
from glob import glob
import os
import math
import sys
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
from pipeline import Pipeline, Molecule, test_data_neighbors
from training_utils import TrainingHistory, train_batch, compute_testing_loss, checkpoint, batch_examples, compare_models
from diagnostics import print_parameter_size, count_parameters, get_object_size
from variable_networks import VariableParityNetwork
from functools import partial
from laurent import LaurentPolynomial
from sparse_kernel_conv import SparseKernelConv, DummyConvolution
if __name__ == '__main__': print("done loading modules.")

if os.name == 'posix':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"\nPreviously: maximum # of open file descriptors: {rlimit[0]} (soft limit) / {rlimit[1]} (hard limit)")
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"\nNow: maximum # of open file descriptors: {rlimit[0]} (soft limit) / {rlimit[1]} (hard limit)")

from collections.abc import Mapping
def sub(c, keys):
    assert isinstance(c, Mapping), f"sub expected a mapping but got a {type(c)}"
    return {k:c[k] for k in keys}

def main():

    ### initialize GPU ###
    device = config.device
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

    ### initialization ###

    print("\n=== Model Generation ===\n")

    # Defaults:
    model_kwargs = {
        'kernel': SparseKernelConv,
        'convolution': DummyConvolution,
        'batch_norm': True,
        }

    load_model_from_file = config.load_model_from_file
    if load_model_from_file:
        model_filenames = glob(load_model_from_file)
        if len(model_filenames) == 0:
            print(f"Could not find any checkpoints matching '{load_model_from_file}'!")
            exit()
        model_filename = max(model_filenames, key = os.path.getctime)
        print(f"Loading model from {model_filename}... ", end='')
        model_dict = torch.load(model_filename)
        model_kwargs.update(model_dict['model_kwargs'])
        all_elements = model_dict['all_elements']
        assert set(all_elements) == set(config.all_elements), "Loaded model elements and config elements don't match!"
    else:
        print("A brand new model was requested... ", end='')
        model_kwargs.update(config.model_kwargs)
        all_elements = config.all_elements

    model = VariableParityNetwork(**model_kwargs)
    print(model_kwargs)
    if load_model_from_file:
        model.load_state_dict(model_dict["state_dict"])
    model.to(device)
    print("Done.")

    Rs_in = config.Rs_in
    Rs_out = config.Rs_out
    max_radius = model_kwargs['max_radius']

    print("Building optimizer... ", end='')
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    if load_model_from_file:
        optimizer.load_state_dict(model_dict["optimizer_state_dict"])
    print("Done.")

    # report configuration
    print("\n=== Configuration ===\n")
    print("all_elements:                     ", all_elements)
    print("relevant_elements:                ", config.relevant_elements)
    print("jiggles_per_molecule:             ", config.jiggles_per_molecule)
    print("testing_size:                     ", config.testing_size)
    print("training_size:                    ", config.training_size)
    print("n_molecule_processors:            ", config.n_molecule_processors)
    print("molecule_queue_max_size:          ", config.molecule_queue_max_size)
    print("data_neighbors_queue_max_size:    ", config.data_neighbors_queue_max_size)
    print("Rs_in:                            ", Rs_in)
    print("Rs_out:                           ", Rs_out)
    print("n_epochs:                         ", config.n_epochs)
    print("batch_size:                       ", config.batch_size)
    print("testing_interval:                 ", config.testing_interval)
    print("checkpoint_interval:              ", config.checkpoint_interval)
    print("checkpoint_prefix:                ", config.checkpoint_prefix)
    print("learning_rate:                    ", config.learning_rate)
    print("muls:                             ", model_kwargs['muls'])
    print("lmaxes:                           ", model_kwargs['lmaxes'])
    print("max_radius:                       ", max_radius)
    print("number_of_basis:                  ", model_kwargs['number_of_basis'])
    print("n_norm:                           ", model_kwargs['n_norm'])

    # print model details
    print("\n=== Model and Optimizer ===\n")
    model_size = get_object_size(model) / 1E6
    optimizer_size = get_object_size(optimizer) / 1E6
    print(f"Model occupies {model_size:.2f} MB; optimizer occupies {optimizer_size:.2f} MB.")
    print("\nParameters per Layer:")
    count_parameters(model)
    print("\nOptimizer:")
    print(optimizer)


    if config.data_source == 'hdf5':
        print(f"Will use training data from {len(config.hdf5_filenames)} files:")
        for filename in config.hdf5_filenames[:4]:
            print(f"   {filename}")
        print("   Etc...")
    elif config.data_source == 'SQL':
        print(f"Using training data from database:")
        print(f"  {config.connect_params['db']}: {config.connect_params['user']}@{config.connect_params['host']}")
        #if 'passwd' not in config.connect_params:
        #    self.connect_params['passwd'] = getpass(prompt="Please enter password: ")

    ### prepare for training ###

    print("\n=== Starting molecule pipeline ===\n")
    print("Working...", end='\r', flush=True)
    relevant_elements = config.relevant_elements
    pipeline = Pipeline(1, max_radius, Rs_in, Rs_out, all_elements, relevant_elements,
            config.jiggles_per_molecule, config.n_molecule_processors, config.molecule_queue_max_size)
    testing_molecules_dict = pipeline.testing_molecules_dict

    print("\n=== Preprocessing Testing Data ===\n")
    print("Working...", end="\r", flush=True)
    time1 = time.time()
    testing_size = config.testing_size
    pipeline.start_reading(testing_size,True,True)  # (how many examples to process,
                                                    #  whether to make Molecules,
                                                    #  whether to save the molecules to a dict)

    # read in and process testing data directly to memory
    testing_examples = []

    print("Reading test examples...")
    while pipeline.any_coming():
        try:
            example = pipeline.get_batch(20)
        except Exception as e:
            print("Failed to get batch!")
            print(e)
            exit()
        testing_examples.append(example)
        #if len(testing_examples) <= 5:
        #    test_data_neighbors(example, Rs_in, Rs_out, max_radius, testing_molecules_dict)
    assert len(testing_examples) == testing_size, \
        f">>>>> expected {testing_size} testing examples but got {len(testing_examples)}"
    
    batch_size = config.batch_size
    print("Batching test examples...")
    testing_batches = batch_examples(testing_examples, batch_size)

    time2 = time.time()
    print(f"Done preprocessing testing data!  That took {time2-time1:.3f} s.\n")
    testing_molecules_dict = dict(testing_molecules_dict)

    ### training ###
    print("\n=== Training ===")
    from training_config import n_epochs, training_size, testing_interval, \
            checkpoint_interval, checkpoint_prefix
    training_history = TrainingHistory()

    for epoch in range(1,n_epochs+1):
        print("                                                                                                ")
        print("Initializing...", end="\r", flush=True)

        # reset the counters for reading the input files
        pipeline.restart()

        # skip over the first testing_size examples
        pipeline.start_reading(testing_size, False, False, batch_size=1, wait=True)
        #pipeline.wait_till_done()

        # process the next training_size examples
        pipeline.start_reading(training_size, True, False, batch_size=batch_size)

        # iterate through all training examples
        minibatches_seen = 0
        #training_data_list = []
        n_minibatches = math.ceil(training_size/batch_size)
        while pipeline.any_coming():
            time1 = time.time()
            data = pipeline.get_batch()

            t_wait = time.time()-time1
            bqsize = pipeline.batch_queue.qsize()

            minibatches_seen += 1
            train_batch(data, model, optimizer, device, training_history)
            #train_batch(data, model2, optimizer2, None)
            training_history.print_training_status_update(epoch, minibatches_seen, n_minibatches, t_wait, 0, bqsize)

            if minibatches_seen % testing_interval == 0:
                #compare_models(model, model2, data, copy_parameters=True)
                compute_testing_loss(model, testing_batches, device, relevant_elements, training_history,
                                     testing_molecules_dict, epoch, minibatches_seen)

            if minibatches_seen % checkpoint_interval == 0:
                checkpoint_filename = f"{checkpoint_prefix}-epoch_{epoch:03d}-checkpoint.torch"
                checkpoint(model_kwargs, model, checkpoint_filename, optimizer, all_elements)
                training_history.write_files(checkpoint_prefix)

    # clean up
    print("                                                                                                 ")
    print("Cleaning up...")
    pipeline.close()

    print("\nProgram complete.")

if __name__ == '__main__':
    freeze_support()
    main()
