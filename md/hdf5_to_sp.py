import cctk, re
import numpy as np
import multiprocessing as mp
import tqdm

#### THIS FILE IS NOT DONE YET

giant_file = "giant_file.xyz"
files = []
new_path = "acetone_10waters_sp-nmr"

num_files = 2000

with open(giant_file, "r+") as f:
    lines = []
    idx = 0
    for line in f:
        if re.search("^$", line):
            lines[0], lines[1] = lines[1], lines[0] # eugene you messed up the xyz format

            filename = f"c_{idx:06d}.xyz"
            with open(filename, "w+") as f2:
                f2.write("".join(lines))
            files.append(filename)

            if idx % 100 == 0:
                print(f"written {idx:06d} files so far")

            lines = []
            idx += 1
        else:
            lines.append(line)

def write(e, idx):
    cctk.GaussianFile.write_ensemble_to_file(
        f"{new_path}_{idx:04d}.gjf",
        e,
        link0={"nprocshared": 4, "mem": "3GB"},
        route_card="#t b3lyp empiricaldispersion=gd3bj gen NMR pop=none int=finegrid nosymm",
        footer="@/n/jacobsen_lab/ekwan/solvent/basis/pcSseg-1.bas",
    )

def read(file):
    mol = cctk.XYZFile.read_file(file).molecule
    mol.assign_connectivity()
    new_mol = mol.limit_solvent_shell(num_solvents=10)
    free_mol = mol.limit_solvent_shell(num_solvents=0)
    return [new_mol, free_mol]

ensembles = [cctk.Ensemble() for i in range(num_files)]

pool = mp.Pool(processes=4)
for i, mol_list in enumerate(tqdm.tqdm(pool.imap(read, files), total=len(files))):
    solv = mol_list[0]
    free = mol_list[1]
    assert isinstance(solv, cctk.Molecule)
    assert isinstance(free, cctk.Molecule)
    ensembles[i % num_files].add_molecule(solv)
    ensembles[i % num_files].add_molecule(free)

for idx, e in enumerate(ensembles):
    write(e, idx)
