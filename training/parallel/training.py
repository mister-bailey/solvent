from training_config import Config
if __name__ != '__main__':
    print("spawning process...")
if __name__ == '__main__': print("loading standard modules...")
import time
from glob import glob
import os
import math
import sys
from datetime import timedelta
from copy import copy
from collections import deque
if __name__ == '__main__': print("loading torch...")
import torch
from torch.multiprocessing import freeze_support
torch.set_default_dtype(torch.float64)
import numpy as np
if __name__ == '__main__': print("loading torch_geometric...")
import torch_geometric as tg
if __name__ == '__main__': print("loading e3nn...")
import e3nn
import e3nn.point.data_helpers as dh
from e3nn.point.message_passing import Convolution
if __name__ == '__main__': print("loading training-specific libraries...")
from pipeline import Pipeline, Molecule, test_data_neighbors, generate_index_shuffle, generate_multi_jiggles_set
from training_utils import train_batch, batch_examples, save_checkpoint, cull_checkpoints, loss_function
from history import TrainTestHistory
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
    config = Config()

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
    model_kwargs.update(config.model.kwargs)
    model = VariableParityNetwork(**model_kwargs)

    save_prefix = config.training.save_prefix
    resume = config.training.resume and save_prefix
    if resume:
        model_filenames = glob(save_prefix + "-*-checkpoint.torch")
        if len(model_filenames) > 0:
            model_filename = max(model_filenames, key = os.path.getmtime)
            print(f"Loading model from {model_filename}... ")
            model_dict = torch.load(model_filename)
            model_kwargs_checkpoint = copy(model_kwargs)
            model_kwargs_checkpoint.update(model_dict['model_kwargs'])
            if model_kwargs != model_kwargs_checkpoint:
                if input("Loaded model doesn't match config file! Build new model and overwrite old? (y/n) ").lower().strip() != 'y': exit()
                resume=False
            else:
                assert set(config.all_elements) == set(model_dict['all_elements']), "Loaded model elements and config elements don't match!"
                config.all_elements = model_dict['all_elements']
                model.load_state_dict(model_dict["state_dict"])
        else:
            print(f"Could not find any checkpoints matching '{save_prefix + '-*-checkpoint.torch'}'!")
            if input("Restart training, OVERWRITING previous training history? (y/n) ").lower().strip() != 'y': exit()
            resume=False
    if not resume:
        print("Building a fresh model... ")
    all_elements = config.all_elements

    print(model_kwargs)
    model.to(device)

    Molecule.initialize_one_hot_table(all_elements)

    #print_parameter_size(model)

    Rs_in = config.model.Rs_in
    Rs_out = config.model.Rs_out
    max_radius = model_kwargs['max_radius']

    print("Building optimizer... ", end='')
    optimizer = torch.optim.Adam(model.parameters(), config.training.learning_rate)
    if resume:
        optimizer.load_state_dict(model_dict["optimizer_state_dict"])
    print("Done.")

    # report configuration
    print("\n=== Configuration ===\n")
    print("all_elements:            ", all_elements)
    print("relevant_elements:       ", config.relevant_elements)
    print("jiggles_per_molecule:    ", config.data.jiggles_per_molecule)
    print("testing_size:            ", config.data.testing_size)
    print("training_size:           ", config.data.training_size)
    print("n_molecule_processors:   ", config.data.n_molecule_processors)
    print("molecule_queue_cap:      ", config.data.molecule_queue_cap)
    print("example_queue_cap:       ", config.data.example_queue_cap)
    print("batch_queue_cap:         ", config.data.batch_queue_cap)
    print("Rs_in:                   ", Rs_in)
    print("Rs_out:                  ", Rs_out)
    print("n_epochs:                ", config.training.n_epochs)
    print("batch_size:              ", config.training.batch_size)
    print("testing_interval:        ", config.training.testing_interval)
    print("save_interval:           ", config.training.save_interval)
    print("save_prefix:             ", save_prefix)
    print("learning_rate:           ", config.training.learning_rate)
    print("muls:                    ", model_kwargs['muls'])
    print("lmaxes:                  ", model_kwargs['lmaxes'])
    print("max_radius:              ", max_radius)
    print("number_of_basis:         ", model_kwargs['number_of_basis'])
    print("n_norm:                  ", model_kwargs['n_norm'])
    print("affine_correction:       ", config.affine_correction)

    # print model details
    print("\n=== Model and Optimizer ===\n")
    model_size = get_object_size(model) / 1E6
    optimizer_size = get_object_size(optimizer) / 1E6
    print(f"Model occupies {model_size:.2f} MB; optimizer occupies {optimizer_size:.2f} MB.")
    print("\nParameters per Layer:")
    count_parameters(model)
    print("\nOptimizer:")
    print(optimizer)


    if config.data.source == 'hdf5':
        print(f"Will use training data from {len(config.data.hdf5_filenames)} files:")
        for filename in config.data.hdf5_filenames[:4]:
            print(f"   {filename}")
        print("   Etc...")
    elif config.data.source == 'SQL':
        print(f"Using training data from database:")
        print(f"  {config.connect_params.db}: {config.connect_params.user}@{config.connect_params.host}")
        #if 'passwd' not in config.connect_params:
        #    self.connect_params['passwd'] = getpass(prompt="Please enter password: ")
    
    ### load or generate test/train shuffle

    testing_size = config.data.testing_size
    training_size = config.data.training_size
    if config.data.randomize and config.data.test_train_shuffle and os.path.isfile(config.data.test_train_shuffle):
        print(f"Loading test/train shuffle indices from {config.data.test_train_shuffle}...")
        test_train_shuffle = torch.load(config.data.test_train_shuffle)
        if len(test_train_shuffle) != testing_size + training_size:
            print(f"Saved test/train shuffle has size {len(test_train_shuffle)}, but config specifies size {testing_size + training_size}!")
            generate_shuffle = True
            if input("Will generate new shuffle. Overwrite old shuffle file? (y/n) ").strip().lower() == "y":
                print("Ok.")
            else:
                config.data.test_train_shuffle = None
                print("Ok. Will discard new shuffle after this run.")
        else:
            generate_shuffle = False 
    else:
        generate_shuffle = True
 
    if generate_shuffle:
        if config.data.randomize:
            print(f"Generating new test/train shuffle from {testing_size + training_size} examples... ", end="")
        else:
            print("Using non-randomized (in-order) test/train indices")
        if not config.data.multi_jiggle_data: # usual database of distinct molecules
            test_train_shuffle = generate_index_shuffle(testing_size + training_size, config.data.connect_params, randomize=config.data.randomize)
        else: # select on smiles string to get specified number of jiggle            
            test_train_shuffle = generate_multi_jiggles_set(
                    math.ceil((testing_size + training_size) / config.data.jiggles_per_molecule), # of molecules
                    config.data.jiggles_per_molecule, config.data.connect_params, config.data.randomize)[
                    :testing_size + training_size]
        print("Done.")
        if config.data.test_train_shuffle and config.data.randomize:
            print(f"Saving test/train shuffle indices to {config.data.test_train_shuffle}...")
            torch.save(test_train_shuffle, config.data.test_train_shuffle)

    #test_set_indices, training_shuffle = test_train_shuffle[:testing_size], test_train_shuffle[testing_size:]    
 
    test_set_indices, training_shuffle = test_train_shuffle[:testing_size], test_train_shuffle[testing_size:]

    ### set up molecule pipeline ###

    print("\n=== Starting molecule pipeline ===\n")
    print("Working...", end='\r', flush=True)
    relevant_elements = config.relevant_elements
    pipeline = Pipeline(config)

    print("\n=== Preprocessing Testing Data ===\n")
    print("Setting test indices...")
    time1 = time.time()
    pipeline.set_indices(test_set_indices)
    pipeline.start_reading(testing_size, batch_size=1)

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
        #    print(f"ID = {example.ID}")
        #    in_out = torch.cat((Molecule.get_atomic_numbers(example.x).unsqueeze(1), example.y), 1)
        #    for a in in_out:
        #        print(a)
        #    print()
        #    test_data_neighbors(example, Rs_in, Rs_out, max_radius, testing_molecules_dict)


    
    batch_size = config.training.batch_size
    print("Batching test examples...")
    testing_batches = batch_examples(testing_examples, batch_size)

    time2 = time.time()
    print(f"Done preprocessing testing data!  That took {time2-time1:.3f} s.\n")
    #testing_molecules_dict = dict(pipeline.testing_molecules_dict)

    ## test/train history ##
    batches_per_epoch = math.ceil(training_size/batch_size)
    if resume:
        try:
            history = TrainTestHistory.load(testing_batches, prefix=save_prefix)
            start_epoch = history.train.current_epoch()
            start_example = history.train.example_in_epoch()
            batch_in_epoch = history.train.next_batch_in_epoch()
            print("Resuming from prior training...")
            print(f"     start_epoch = {start_epoch}")
            print(f"   start_example = {start_example}")
            print(f"  batch_in_epoch = {batch_in_epoch}")
            partial_epoch = True
        except Exception as e:
            print(f"Failed to load history from {save_prefix + '-history.torch'}")
            print(e)
            if input("Continue training with old model but new training history? (y/n) ").lower().strip() != 'y':
                print("Exiting...")
                exit()
            resume = False
    if not resume:
        history = TrainTestHistory(training_size, save_prefix, testing_batches,
                               relevant_elements, device, config.number_to_symbol)
        partial_epoch = False
        start_epoch = 1

    ### training ###
    print("\n=== Training ===")
    n_epochs = config.training.n_epochs
    time_limit = config.training.time_limit
    testing_interval = config.training.testing_interval
    save_interval = config.training.save_interval
    num_checkpoints = config.training.num_checkpoints

    pipeline.set_indices(training_shuffle)

    start_elapsed = history.elapsed_time()
    data_queue = deque(maxlen=config.data.batch_preload)

    for epoch in range(start_epoch, start_epoch + n_epochs):
        print("                                                                                                ")
        print(("Resuming" if partial_epoch else "Starting") + f" epoch {epoch}...", end="\r", flush=True)

        # though we may start partway through an epoch, subsequent epochs start at example 0 and batch 1
        if partial_epoch:
            partial_epoch = False
        else:
            start_example = 0
            batch_in_epoch = 1

        # return to the start of the training set
        pipeline.scan_to(start_example)

        # process the training examples
        pipeline.start_reading(training_size - start_example, batch_size=batch_size)

        # iterate through all training examples    
        while pipeline.any_coming() or len(data_queue) > 0:

            time1 = time.time()
            while pipeline.any_coming() and len(data_queue) < config.data.batch_preload:
                data = pipeline.get_batch().to(device)
                data_queue.appendleft(data)
            t_wait = time.time()-time1

            #if epoch == start_epoch and batch_in_epoch == 1:
            #    print("\nTesting positions:")
            #    print(testing_batches[0].pos)
            #    print("\nTraining positions:")
            #    print(data.pos)


            batch_loss, train_time = train_batch(data_queue, model, optimizer)#, device)

            # To keep training loss up to date with testing loss
            #if batch_in_epoch % testing_interval == 0 or batch_in_epoch == batches_per_epoch:
            #    model.eval()
            #    with torch.no_grad():
            #        data = data.to(device)
            #        output = model(data.x, data.edge_index, data.edge_attr)
            #        loss, _ = loss_function(output,data)
            #    batch_loss = loss.sqrt().item()
            
            history.train.log_batch(train_time, t_wait, data.n_examples, len(data.x), batch_loss, epoch=epoch)

            if batch_in_epoch % testing_interval == 0 or not pipeline.any_coming():
                history.run_test(model)

            times_up = time_limit is not None and history.elapsed_time() - start_elapsed >= time_limit

            if batch_in_epoch % save_interval == 0 or not pipeline.any_coming() or times_up:
                checkpoint_filename = f"{save_prefix}-e{epoch:03d}_b{batch_in_epoch:05}-checkpoint.torch"
                save_checkpoint(model_kwargs, model, checkpoint_filename, optimizer, all_elements)
                history.save()
                if num_checkpoints:
                    cull_checkpoints(save_prefix, num_checkpoints)
            
            if times_up:
                print("                                                                                                 ")
                print(f"Finished training for {str(timedelta(seconds=history.elapsed_time() - start_elapsed))[:-5]}.")
                print("Cleaning up...")
                pipeline.close()
                print("\nProgram complete.")
                exit()

            batch_in_epoch += 1

    # clean up
    print("                                                                                                 ")
    print(f"Finished training {n_epochs} epochs.")
    print("Cleaning up...")
    pipeline.close()

    print("\nProgram complete.")

if __name__ == '__main__':
    freeze_support()
    main()
