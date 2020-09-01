from training_config import Config
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
torch.set_default_dtype(torch.float64)
if __name__ == '__main__': print("loading training-specific libraries...")
from pipeline import Pipeline, Molecule, test_data_neighbors, generate_index_shuffle, generate_multi_jiggles_set
from training_utils import train_batch, batch_examples, save_checkpoint, cull_checkpoints
if __name__ == '__main__': print("done loading modules.")

def main():
    config = Config()

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
            print(f"Generating new test indices from {testing_size} examples... ", end="")
        else:
            print("Using non-randomized (in-order) test/train indices")
        if not config.data.multi_jiggle_data: # usual database of distinct molecules
            test_train_shuffle = generate_index_shuffle(testing_size, config.data.connect_params, randomize=config.data.randomize)
        else: # select on smiles string to get specified number of jiggle            
            test_train_shuffle = generate_multi_jiggles_set(
                    math.ceil((testing_size + training_size) / config.data.jiggles_per_molecule), # of molecules
                    config.data.jiggles_per_molecule, config.data.connect_params, config.data.randomize)[
                    :testing_size + training_size]
        print("Done.")
        if config.data.test_train_shuffle and config.data.randomize:
            print(f"Saving test/train shuffle indices to {config.data.test_train_shuffle}...")
            torch.save(test_train_shuffle, config.data.test_train_shuffle)

    test_set_indices = test_train_shuffle[:testing_size]     

    #print("Test set indices:")
    #print(test_set_indices[:100], "...")

    ### set up molecule pipeline ###

    print("\n=== Starting molecule pipeline ===\n")
    print("Working...", end='\r', flush=True)
    pipeline = Pipeline(config, new_process=False)
    testing_molecules_dict = pipeline.testing_molecules_dict

    print("\n=== Processing test data ===\n")
    print("Setting test indices...")
    time1 = time.time()
    pipeline.set_indices(test_set_indices)
    print("calling dataset_reader.run()...")
    pipeline.dataset_reader.run()
    print("Resetting database pointer...")
    pipeline.dataset_reader.run()

    print("\n=== Reading into pipeline ===\n")
    pipeline.start_reading(testing_size, batch_size=1, record_in_dict=True)

    pipeline.dataset_reader.run()

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
    
    batch_size = config.training.batch_size
    print("Batching test examples...")
    testing_batches = batch_examples(testing_examples, batch_size)

    time2 = time.time()
    print(f"Done preprocessing testing data!  That took {time2-time1:.3f} s.\n")
    testing_molecules_dict = dict(testing_molecules_dict)



if __name__ == '__main__':
    main()
