import cctk, re, tqdm, h5py, sys, os, yaml, logging
import numpy as np
import multiprocessing as mp

#### This script creates NMR jobs from a trajectory saved as an ``.hdf5`` file in the MDTraj format.

#### USAGE:
#### python hdf5_to_sp.py save.hdf5 path/to/nmr_prefix

#### Expects ``config.yml`` in the same directory.

logging.basicConfig(level=logging.INFO, filename="log.log", format='%(asctime)s %(name)-12s  %(message)s', datefmt='%m-%d %H:%M', filemode="w")
logger = logging.getLogger(__name__)

hdf5_path = sys.argv[1]
new_path = sys.argv[2]
logger.info(f"Reading from {hdf5_path}")
logger.info(f"Writing to {new_path}_xxxx.gjf")

config = yaml.load(open("config.yml", "r"), yaml.Loader)
num_files = config["num_nmr_files"]

def write(m, idx):
    assert isinstance(m, cctk.Molecule), "need a ``cctk`` molecule to write!"
    filename = f"{new_path}_{idx:04d}.gjf"
    if not os.path.exists(filename):
        cctk.GaussianFile.write_molecule_to_file(
            f"{new_path}_{idx:04d}.gjf",
            m,
            link0={"nprocshared": 4, "mem": "3GB"},
            route_card=config["route_card"]
            footer=config["footer"]
        )
    else:
        cctk.GaussianFile.write_molecule_to_file(
            f"{new_path}_{idx:04d}.gjf",
            m,
            link0={"nprocshared": 4, "mem": "3GB"},
            route_card=config["route_card"]
            footer=config["footer"]
            append=True,
        )

def read(args):
    coords = args[0] * 10 # nm to Å
    atomic_numbers = args[1]
    length = args[2]

    #### move first atom to middle and then move PBCs
    coords += -1 * coords[0]
    coords += length / 2    
    coords = np.mod(coords, length)

    mol = cctk.Molecule(atomic_numbers, coords)
    mol.assign_connectivity()

    new_mol = mol.limit_solvent_shell(num_solvents=config["num_solvents"])
    free_mol = mol.limit_solvent_shell(num_solvents=0)
    return [new_mol, free_mol]

counter = 0
idx = 1

#### we multithread the file reading since it's SLOW
pool = mp.Pool(processes=config["num_threads"])
with h5py.File(hdf5_path, "r") as h5:
    coords = h5["coordinates"]
    per_file = int(len(coords) / num_files)
    logger.info(f"Will write {num_files} files with {per_file} * 2 structures per file -- {len(coords)} * 2 structures in total")

    topology_str = h5["topology"][0].decode()
    length = h5["cell_lengths"][0][0] * 10 # nm to Å

    logger.info(f"Periodic boundary conditions - cube of side length {length:.2f} Å")

    atomic_symbols = re.findall(r'"element": "(?P<element>[A-Z][a-z]?)"', topology_str)
    atomic_numbers = np.array(list(map(cctk.helper_functions.get_number, atomic_symbols)), dtype=np.int8)

    args = [(c, atomic_numbers, length) for c in coords]

    for i, mol_list in enumerate(tqdm.tqdm(pool.imap(read, args), total=len(coords))):
        write(mol_list[0], idx)
        write(mol_list[1], idx)

        if counter % per_file == 0:
            idx += 1
            counter = 1
        else:
            counter += 1

logger.info("Finished writing files")
