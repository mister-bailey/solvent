if __name__ == '__main__': print("Loading numpy...")
import numpy as np
if __name__ == '__main__': print("Loading torch...")
import torch
torch.set_default_dtype(torch.float64)
if __name__ == '__main__': print("Loading torch.multiprocessing...")
import torch.multiprocessing as mp
if __name__ == '__main__': print("Loading torch.distributed...")
import torch.distributed as dist
if __name__ == '__main__': print("Loading DistributedDataParallel...")
from torch.nn.parallel import DistributedDataParallel
if __name__ == '__main__': print("Loading e3nn...")
import e3nn
import torch_geometric as tg
if __name__ == '__main__': print("Loading time...")
import time
from collections.abc import Mapping
if __name__ == '__main__': print("Loading sparse_kernel_conv...")
from sparse_kernel_conv import SparseKernelConv, DummyConvolution
if __name__ == '__main__': print("Loading laurent...")
from laurent import LaurentPolynomial
if __name__ == '__main__': print("Loading functools...")
from functools import partial
if __name__ == '__main__': print("Loading variable_networks...")
from variable_networks import VariableParityNetwork
if __name__ == '__main__': print("Loading diagnostics...")
from diagnostics import print_parameter_size, count_parameters, get_object_size
if __name__ == '__main__': print("Loading history...")
from history import TrainTestHistory
if __name__ == '__main__': print("Loading collections...")
from collections import deque
if __name__ == '__main__': print("Loading copy...")
from copy import copy
if __name__ == '__main__': print("Loading datetime...")
from datetime import timedelta
if __name__ == '__main__': print("Loading re...")
import re
if __name__ == '__main__': print("Loading sys...")
import sys
if __name__ == '__main__': print("Loading os...")
import os
if __name__ == '__main__': print("Loading math...")
import math
if __name__ == '__main__': print("Loading glob...")
from glob import glob
if __name__ == '__main__': print("Loading training_config...")
from training_config import Config
if __name__ == '__main__': print("Loading training_utils...")
from training_utils import train_batch, batch_examples, save_checkpoint, cull_checkpoints, loss_function
if __name__ == '__main__': print("Loading pipeline...", flush=True)
from pipeline import Pipeline, Molecule, test_data_neighbors, generate_index_shuffle, generate_multi_jiggles_set

if __name__ != '__main__':
    print("spawning process...")

if __name__ == '__main__':
    print("done loading modules.")

if os.name == 'posix' and __name__ == '__main__':
    mp.set_start_method('spawn')
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(
        f"\nPreviously: maximum # of open file descriptors: {rlimit[0]} (soft limit) / {rlimit[1]} (hard limit)")
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(
        f"\nNow: maximum # of open file descriptors: {rlimit[0]} (soft limit) / {rlimit[1]} (hard limit)")


def sub(c, keys):
    assert isinstance(
        c, Mapping), f"sub expected a mapping but got a {type(c)}"
    return {k: c[k] for k in keys}


def main():
    config = Config()

    ### initialize GPU ###
    device = config.device
    if device == "cuda":
        device = "cuda:0"
    print("\n=== GPU settings: ===\n")
    print(f"current cuda device: {torch.cuda.current_device()}")
    print(f"cuda device count:   {torch.cuda.device_count()}")
    print(f"cuda device name:    {torch.cuda.get_device_name(0)}")
    print(f"is cuda available?   {torch.cuda.is_available()}")
    print(f"cuda version:        {torch.version.cuda}")
    print(f"device:              {device}")
    # print(torch.cuda.memory_summary())
    temp_tensor = torch.rand(10).to(device)
    print("test tensor:")
    print(temp_tensor)

    if config.parallel and not dist.is_available():
        print(
            "Configured for distributed computing, but that's not available on this system!")
        if not input("  Proceed with single GPU training? (y/n)").strip().lower() == 'y':
            exit()
        else:
            config.parallel = False

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
            model_filename = max(model_filenames, key=os.path.getmtime)
            print(f"Loading model from {model_filename}... ")
            model_dict = torch.load(model_filename)
            model_kwargs_checkpoint = copy(model_kwargs)
            model_kwargs_checkpoint.update(model_dict['model_kwargs'])
            if model_kwargs != model_kwargs_checkpoint:
                if input("Loaded model doesn't match config file! Build new model and overwrite old? (y/n) ").lower().strip() != 'y':
                    exit()
                resume = False
            else:
                assert set(config.all_elements) == set(
                    model_dict['all_elements']), "Loaded model elements and config elements don't match!"
                config.all_elements = model_dict['all_elements']
                model.load_state_dict(model_dict["state_dict"])
        else:
            print(
                f"Could not find any checkpoints matching '{save_prefix + '-*-checkpoint.torch'}'!")
            if input("Restart training, OVERWRITING previous training history? (y/n) ").lower().strip() != 'y':
                exit()
            resume = False
    if not resume:
        print("Building a fresh model... ")
    all_elements = config.all_elements

    print(model_kwargs)
    model.to(device)

    Molecule.initialize_one_hot_table(all_elements)

    # print_parameter_size(model)

    Rs_in = config.model.Rs_in
    Rs_out = config.model.Rs_out
    max_radius = model_kwargs['max_radius']

    print("Building optimizer... ", end='')
    optimizer = torch.optim.Adam(
        model.parameters(), config.training.learning_rate)
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
    print(
        f"Model occupies {model_size:.2f} MB; optimizer occupies {optimizer_size:.2f} MB.")
    print("\nParameters per Layer:")
    count_parameters(model)
    print("\nOptimizer:")
    print(optimizer)

    if config.data.source == 'hdf5':
        print(
            f"Will use training data from {len(config.data.hdf5_filenames)} files:")
        for filename in config.data.hdf5_filenames[:4]:
            print(f"   {filename}")
        print("   Etc...")
    elif config.data.source == 'SQL':
        print(f"Using training data from database:")
        print(
            f"  {config.connect_params.db}: {config.connect_params.user}@{config.connect_params.host}")
        # if 'passwd' not in config.connect_params:
        #    self.connect_params['passwd'] = getpass(prompt="Please enter password: ")

    # load or generate test/train shuffle

    testing_size = config.data.testing_size
    training_size = config.data.training_size
    if config.data.randomize and config.data.test_train_shuffle and os.path.isfile(config.data.test_train_shuffle):
        print(
            f"Loading test/train shuffle indices from {config.data.test_train_shuffle}...")
        test_train_shuffle = torch.load(config.data.test_train_shuffle)
        if len(test_train_shuffle) != testing_size + training_size:
            print(
                f"Saved test/train shuffle has size {len(test_train_shuffle)}, but config specifies size {testing_size + training_size}!")
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
            print(
                f"Generating new test/train shuffle from {testing_size + training_size} examples... ", end="")
        else:
            print("Using non-randomized (in-order) test/train indices")
        if not config.data.multi_jiggle_data:  # usual database of distinct molecules
            test_train_shuffle = generate_index_shuffle(
                testing_size + training_size, config.data.connect_params, randomize=config.data.randomize)
        else:  # select on smiles string to get specified number of jiggle
            test_train_shuffle = generate_multi_jiggles_set(
                math.ceil((testing_size + training_size) /
                          config.data.jiggles_per_molecule),  # of molecules
                config.data.jiggles_per_molecule, config.data.connect_params, config.data.randomize)[
                :testing_size + training_size]
        print("Done.")
        if config.data.test_train_shuffle and config.data.randomize:
            print(
                f"Saving test/train shuffle indices to {config.data.test_train_shuffle}...")
            torch.save(test_train_shuffle, config.data.test_train_shuffle)

    #test_set_indices, training_shuffle = test_train_shuffle[:testing_size], test_train_shuffle[testing_size:]

    test_set_indices, training_shuffle = test_train_shuffle[:
                                                            testing_size], test_train_shuffle[testing_size:]

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
        #    print(f"batch.y : {list(example.y.shape)}")
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
    print(f"Done preprocessing testing data.  That took {time2-time1:.3f} s.\n")

    preload = config.data.batch_preload
    dist_batch_size = batch_size
    if config.parallel:  # and config.gpus > 1:
        config.gpus = min(config.gpus, torch.cuda.device_count())
        print(f"\n === Setting up parallel training on {config.gpus} GPUs ===")

        # choose a big tensor to compare across processes for testing:
        test_key = None
        test_distrib = False
        if test_distrib:
            test_size = 0
            for key, tensor in model.state_dict().items():
                if tensor.numel() > test_size:
                    test_key = key
                    test_size = tensor.numel()
            test_tensor = model.state_dict()[test_key].clone().detach()
            test_key = 'module.' + test_key # wrapped in DistributedDataParallel
            print(f"Using model.state_dict()['{test_key}'] for inter-GPU testing.'")
            
        print(f"Spawning {config.gpus-1} extra processes...", flush=True)
        worker_pool = mp.spawn(aux_train, (config.gpus, pipeline,
                config.training.learning_rate, model_kwargs, None, #model.state_dict(),
                optimizer.state_dict(), preload, test_key),  config.gpus-1, join=False)
                
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'  
        print("Initializing process group...")
        dist.init_process_group(backend="nccl", rank=0, world_size=config.gpus) 
        print("Building DistributedDataParallel model... ", flush=True)
        torch.cuda.set_device(0)
        model = DistributedDataParallel(model, device_ids=[0], output_device=0)
        dist.barrier()
        dist_batch_size = round(batch_size / config.gpus)
        batch_size = dist_batch_size * config.gpus

    ## test/train history ##
    print("\n === Setting up logging and testing ===")

    run_name = re.findall(r"[^\\/]+", save_prefix)[-2]
    save_dir = os.path.join(*(re.findall(r"[^\\/]+", save_prefix)[:-1]))
    print(f"Run name: {run_name}")

    use_wandb = config.training.use_wandb
    if use_wandb:
        import wandb
        wandb.login(key=config.wandb.pswd)
        
        from configparser import ConfigParser
        wb_cfg = ConfigParser()
        wb_cfg.read(os.path.join(save_dir, "wandb.ini"))
        if 'wandb' not in wb_cfg:
            wb_cfg.add_section('wandb')
        if not resume or 'id' not in wb_cfg['wandb']:
            wb_cfg['wandb']['id'] = wandb.util.generate_id()
        with open(os.path.join(save_dir, "wandb.ini"), 'w') as wb_configfile:
            wb_cfg.write(wb_configfile)
            
        if 'id' in wb_cfg['wandb']:
            wandb.init(name=run_name, project=config.project, dir=save_dir, resume='allow', id=wb_cfg['wandb']['id'])
        else:
            wandb.init(name=run_name, project=config.project, dir=save_dir, resume='allow')

        wandb.config.batch_size = config.training.batch_size
        wandb.config.learning_rate = config.training.learning_rate
        wandb.config.muls = config.model.muls
        wandb.config.lmaxes = config.model.lmaxes
        wandb.config.max_radius = config.model.max_radius
        wandb.config.number_of_basis = config.model.number_of_basis
        wandb.config.radial_h = config.model.radial_h
        wandb.config.radial_layers = config.model.radial_layers
        wandb.config.n_norm = config.model.n_norm
        wandb.config.batch_norm = config.model.batch_norm
        wandb.config.batch_norm_momentum = config.model.batch_norm_momentum
        wandb.config.use_tensor_constraint = config.training.use_tensor_constraint

    wandb_log = wandb.log if use_wandb else None
    use_tensor_constraint = config.training.use_tensor_constraint

    if resume:
        try:
            history = TrainTestHistory.load(
                testing_batches, prefix=save_prefix, wandb_log=wandb_log)
            start_epoch = history.train.current_epoch()
            example_number = history.train.example_in_epoch()
            batch_in_epoch = history.train.next_batch_in_epoch()
            print("Resuming from prior training...")
            print(f"     start_epoch = {start_epoch}")
            print(f"   start_example = {example_number}")
            print(f"  batch_in_epoch = {batch_in_epoch}")
            partial_epoch = True
        except Exception as e:
            print(
                f"Failed to load history from {save_prefix + '-history.torch'}")
            print(e)
            if input("Continue training with old model but new training history? (y/n) ").lower().strip() != 'y':
                print("Exiting...")
                exit()
            resume = False
    if not resume:
        history = TrainTestHistory(training_size, batch_size, testing_batches, relevant_elements, device,
                                   save_prefix, run_name, config.number_to_symbol,
                                   use_tensor_constraint=use_tensor_constraint, sparse_logging=True,
                                   wandb_log=wandb_log)
        partial_epoch = False
        start_epoch = 1

    ### training ###
    print("\n=== Training ===")
    n_epochs = config.training.n_epochs
    time_limit = config.training.time_limit
    testing_interval = config.training.testing_interval
    save_interval = config.training.save_interval
    num_checkpoints = config.training.num_checkpoints
    use_tensor_constraint = config.training.use_tensor_constraint

    pipeline.set_indices(training_shuffle)

    start_elapsed = history.elapsed_time()
    
    test_index = 0

    data_queue = deque(maxlen=preload)
    for epoch in range(start_epoch, start_epoch + n_epochs):
        print("                                                                                                ")
        print(("Resuming" if partial_epoch else "Starting") +
              f" epoch {epoch}...", end="\r", flush=True)

        # though we may start partway through an epoch, subsequent epochs start at example 0 and batch 1
        if partial_epoch:
            partial_epoch = False
        else:
            example_number = 0
            batch_in_epoch = 1

        batch_of_last_test = (
            batch_in_epoch // testing_interval) * testing_interval
        batch_of_last_save = (batch_in_epoch // save_interval) * save_interval

        # return to the start of the training set
        #print("Scanning to start of training set...")
        pipeline.scan_to(example_number)

        # process the training examples
        #print("Starting read into pipeline...")
        pipeline.start_reading(
            training_size - example_number, batch_size=dist_batch_size)

        # iterate through all training examples
        while pipeline.any_coming() or len(data_queue) > 0:

            time1 = time.time()
            while len(data_queue) < preload and pipeline.any_coming():
                data = pipeline.get_batch().to(device)
                data_queue.appendleft(data)
            t_wait = time.time()-time1

            time1 = time.time()
            train_losses = train_batch(data_queue, model, optimizer, use_tensor_constraint=use_tensor_constraint)
            train_time = time.time() - time1

            example_number = min(example_number + batch_size, training_size)
            batch_in_epoch += 1
            
            history.train.log_batch(train_time, t_wait, data.n_examples, len(
                data.x), example_number, *train_losses, epoch=epoch)

            if config.parallel and test_key is not None and test_index < 10:
                test_rank = (test_index % (config.gpus-1)) + 1
                test_index += 1
                dist.barrier()
                print(f"\n[0]: receiving test params from rank {test_rank}...", flush=True)
                dist.broadcast(test_tensor, src=test_rank)
                if torch.equal(test_tensor, model.state_dict()[test_key]):
                    print("  Model is synchronized!", flush=True)
                else:
                    print("  Not synchronized!", flush=True)

            if batch_in_epoch - batch_of_last_test >= testing_interval or not pipeline.any_coming():
                batch_of_last_test = batch_in_epoch
                history.run_test(model)

            times_up = time_limit is not None and history.elapsed_time() - \
                start_elapsed >= time_limit

            if batch_in_epoch - batch_of_last_save >= save_interval or not pipeline.any_coming() or times_up:
                batch_of_last_save = batch_in_epoch
                checkpoint_filename = f"{save_prefix}-e{epoch:03d}_b{batch_in_epoch:05}-checkpoint.torch"
                save_checkpoint(model_kwargs, model,
                                checkpoint_filename, optimizer, all_elements)
                history.save()
                if num_checkpoints:
                    cull_checkpoints(save_prefix, num_checkpoints)

            if times_up:
                print(
                    "                                                                                                 ")
                print(
                    f"Finished training for {str(timedelta(seconds=history.elapsed_time() - start_elapsed))[:-5]}.")
                print("Cleaning up...")
                pipeline.close()
                print("\nProgram complete.")
                exit()

    # clean up
    print("                                                                                                 ")
    print(f"Finished training {n_epochs} epochs.")
    print("Cleaning up...")
    pipeline.close()

    print("\nProgram complete.")

# auxiliary training processes (not the main one)
def aux_train(rank, world_size, pipeline, learning_rate, model_kwargs, model_state_dict=None, optimizer_state_dict=None, preload=1, test_key=None):
    rank += 1  # We already have a "main" process which should take rank 0
    print(f"\nGPU rank {rank} reporting for duty!\n")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)#, group_name="train")

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    model = VariableParityNetwork(**model_kwargs)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    dist.barrier()
    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    test_index = 0
    verbose = False #True if rank==1 else False
    
    if verbose: print(f"[{rank}]: pipeline.batch_queue = {pipeline.batch_queue}", flush=True)

    data_queue = deque(maxlen=preload)
    while True:
        if verbose: print(f"[{rank}]: Starting train loop...", flush=True)
        #if verbose: print(f"[{rank}]: Querying pipeline... ", end='', flush=True)
        #any_coming = pipeline.any_coming()
        #if verbose: print(f"Done. ", flush=True)
        time1 = time.time()
        while (len(data_queue) < preload and pipeline.any_coming()) or len(data_queue)==0:
            if verbose: print(f"[{rank}]: Getting batch... ", end='', flush=True)
            data = pipeline.get_batch().to(device)
            data_queue.appendleft(data)
            if verbose: print("Done.", flush=True)
        wait_time = time.time() - time1
        if verbose: print(f"[{rank}]: wait time = {wait_time:.2f}", flush=True)

        if verbose: print(f"[{rank}]: Training batch...", flush=True)
        _ = train_batch(data_queue, model, optimizer)
        if verbose: print(f"[{rank}]: Finished training batch.", flush=True)

        if test_key is not None and test_index < 10:
            test_rank = (test_index % (world_size-1)) + 1
            test_index += 1
            dist.barrier()
            if verbose and test_rank == rank: print(f"\n[{rank}]: sending test params ... ", flush=True)  
            dist.broadcast(model.state_dict()[test_key], src=test_rank)

    


if __name__ == '__main__':
    mp.freeze_support()
    main()
