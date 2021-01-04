import numpy as np
import h5py, re, ast
from glob import glob

# all expected elements
all_elements = ['C', 'H', 'N', 'O', 'S']
n_elements = len(all_elements)

# so we can normalize training data for the nuclei to be predicted
elementwide_scaling_factors = { "C" : (118.0, 51.0),    # element : (mean, stdev)
                                "H" : (29.0, 2.0)  }
relevant_elements = list(elementwide_scaling_factors.keys())

# irrelevant atoms will not be scaled
for element in all_elements:
    if element not in elementwide_scaling_factors:
        elementwide_scaling_factors[element] = (0.0,1.0)

# create a function to noramlize the shieldings for a specific molecule
# shielding -> (shielding-mean)/stdev
def get_scalings(atomic_symbols):
    scaling_factors = [ elementwide_scaling_factors[a] for a in atomic_symbols ]
    scaling_factors = np.array(scaling_factors)
    def scaling_function(x):
        result = x - scaling_factors[:,0]
        result = result / scaling_factors[:,1]
        return result
    return scaling_factors, scaling_function

# generates one-hots for a list of atomic_symbols
def get_one_hots(atomic_symbols):
    one_hots = []
    for symbol in atomic_symbols:
        inner_list = [ 1. if symbol == i else 0. for i in all_elements ]
        one_hots.append(inner_list)
    return np.array(one_hots)

# compute weights for loss function
def get_weights(atomic_symbols, symmetrical_atoms):
    weights = [ 1.0 if symbol in relevant_elements else 0.0 for symbol in atomic_symbols ]
    weights = np.array(weights)
    for l in symmetrical_atoms:
        weight = 1.0/len(l)
        for i in l:
            weights[i] = weight
    return weights

class Molecule():
    def __init__(self, name,
                 atomic_symbols,
                 symmetrical_atoms,        # list of lists of 0-indexed atom numbers
                 perturbed_geometries,
                 stationary_shieldings,
                 perturbed_shieldings):
        self.name = name                                                     # name of molecule
        self.atomic_symbols = atomic_symbols                                 # vector of strings of length n_atoms
        self.n_atoms = len(atomic_symbols)                                   # number of atoms
        self.perturbed_geometries = perturbed_geometries                     # (n_examples, n_atoms, 3)
        self.stationary_shieldings = stationary_shieldings                   # (n_examples, n_atoms, 1)
        scaling_factors, scaling_function = get_scalings(atomic_symbols)
        self.scaling_factors = scaling_factors                               # (n_atoms, 2) inner axis is mean, stdev
        self.scaling_function = scaling_function                             # unscaled shieldings --> scaled shieldings 

        # scale perturbed shieldings and zero out data for irrelevant atoms
        perturbed_shieldings = self.scaling_function(perturbed_shieldings)
        for i,a in enumerate(atomic_symbols):          # zero out shieldings
            if a not in relevant_elements:             # for irrelevant
                perturbed_shieldings[:,i]=0.0          # elements
        self.perturbed_shieldings = perturbed_shieldings                     # (n_examples, n_atoms, 1) 
        self.features = get_one_hots(atomic_symbols)                         # (n_atoms, n_elements)
        self.weights = get_weights(atomic_symbols, symmetrical_atoms)        # (n_atoms,)

def str2array(s):
    # https://stackoverflow.com/questions/35612235/how-to-read-numpy-2d-array-from-string
    s = re.sub('\[ +', '[', s.strip())
    s = re.sub('[,\s]+', ', ', s)
    a = ast.literal_eval(s)
    if len(a) == 0 or a is None:
        return []
    else:
        for i, b in enumerate(a):
            for j, _ in enumerate(b):
                 a[i][j] += -1

        return a

# read Molecule data from a group of hdf5 files
# returns molecule_dict: {name : Molecule}
def read_molecule_data(filenames, max_molecules=None):
    molecules_dict = {}
    total_num = 0

    for filename in filenames:
        print(f"Reading data from {filename}...")
        with h5py.File(filename, "r") as h5:
            for dataset_name,geometries_and_shieldings in h5.items():
                if not dataset_name.startswith("data_"):
                    continue
                dataset_number = dataset_name.split("_")[1]
                n_molecules = len(molecules_dict)+1
                print(f"  Processing molecule {n_molecules} ({dataset_name})...", end='\r', flush=True)
                geometries_and_shieldings = np.array(geometries_and_shieldings)
                perturbed_geometries = geometries_and_shieldings[:,:,:3]
                perturbed_shieldings = geometries_and_shieldings[:,:,3]
                assert np.shape(geometries_and_shieldings)[2] == 4
                stationary_shieldings = h5.attrs[f"stationary_{dataset_number}"]
                n_atoms = len(stationary_shieldings)
                assert np.shape(geometries_and_shieldings)[1] == n_atoms, f"expected {n_atoms} atoms, but got {np.shape(geometries_and_shieldings)[1]}"
                atomic_symbols = h5.attrs[f"atomic_symbols_{dataset_number}"]
                for a in atomic_symbols:
                    assert a in all_elements, f"unexpected element!  need to add {a} to all_elements"
                assert len(atomic_symbols) == n_atoms, f"expected {n_atoms} atoms, but got {len(atomic_symbols)}"
                symmetrical_atoms = str2array(h5.attrs[f"symmetrical_atoms_{dataset_number}"])

                total_num += len(perturbed_shieldings)

                # store the results
                molecule = Molecule(dataset_number, atomic_symbols, symmetrical_atoms,
                                    perturbed_geometries, stationary_shieldings, perturbed_shieldings)
                molecules_dict[dataset_number] = molecule
                if n_molecules == max_molecules:
                    break

        print(f"Done!  {len(molecules_dict)} molecules and {total_num} training examples read so far.\n")
    print("Reading is complete.")
    return molecules_dict

