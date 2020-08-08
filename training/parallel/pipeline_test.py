from pipeline import *
import training_config
import sys
from torch.multiprocessing import freeze_support


def main():
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
    max_radius = training_config.max_radius


    pipeline = Pipeline(hdf5_filenames, 1, max_radius, Rs_in, Rs_out, jiggles_per_molecule,
            n_molecule_processors, molecule_queue_max_size, new_process=False)
    testing_molecules_dict = pipeline.testing_molecules_dict
    pipeline.start_reading(testing_size,True,True)  # (how many examples to process,
                                                    #  whether to make Molecules,
                                                    #  whether to save the molecules to a dict)
    pipeline.dataset_reader.run()

    

    # read in and process testing data directly to memory
    testing_data_list = []

    while pipeline.any_coming():
        try:
            example = pipeline.get_batch(20)
        except Exception as e:
            print("Failed to get batch!")
            print(e)
            sys.exit()
        print(f"Got test example {len(testing_data_list)}")
        testing_data_list.append(example)
    assert len(testing_data_list) == testing_size, \
        f">>>>> expected {testing_size} testing examples but got {len(testing_data_list)}"



if __name__ == '__main__':
    freeze_support()
    main()
