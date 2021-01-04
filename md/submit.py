import argparse, math, sys, os, subprocess, glob, re, shutil, logging

#### This script runs MD for a cube of solvent and automatically extract frames for NMR.

#### USAGE:
#### python submit.py "SMILES" name_of_molecule

#### OPTIONS
#### -p - partition name
#### -c - num cores
#### -g - num gpu cores
#### -t - time in minutes
#### -m - mem per core (MB)
#### -F - remove directory, if it already exists

#### Requires a copy of ``run_from_smiles.py`` and ``mdconf.yml`` in current directory.
#### Also, a conda environment named ``openmm``.
#### Corin Wagen, 2020

parser = argparse.ArgumentParser(prog="submit.py")
parser.add_argument("--partition", "-p", type=str, default="gpu")
parser.add_argument("--config", "-c", type=str, default="config.yml")
parser.add_argument("--time", "-t", type=int, default=7200)
parser.add_argument("--cores", "-n", type=int, default=16)
parser.add_argument("--gpu_cores", "-g", type=int, default=2)
parser.add_argument("--mem", "-m", type=int, default=2000) # per core, in MB
parser.add_argument("--force", "-F", default=False, action="store_true")
parser.add_argument("smiles", type=str)
parser.add_argument("name", type=str)

args = vars(parser.parse_args(sys.argv[1:]))

name = args["name"]

if os.path.exists(name):
    if args["force"]:
        print(f"removing {name}/")
        shutil.rmtree(name)
    else:
        print(f"{name}/ already exists -- use the '-F' flag to remove!")

os.mkdir(name)
shutil.copyfile(args["config"], f"{name}/config.yml")
shutil.copyfile("run_from_smiles.py", f"{name}/run_from_smiles.py")
shutil.copyfile("hdf5_to_sp.py", f"{name}/hdf5_to_sp.py")

os.chdir(name)
print(f"{name}/ created")
os.mkdir("inputs")
print(f"{name}/inputs/ created")

text = "#!/bin/bash\n"
text += "#SBATCH -N 1\n"
text += f"#SBATCH -n {args['cores']}\n"
if re.search("gpu", args["partition"]):
    text += f"#SBATCH --gres=gpu:{args['gpu_cores']}\n"
text += f"#SBATCH -p {args['partition']}\n"
text += f"#SBATCH --mem={args['cores']*args['mem'] + 2000}\n"
text += f"#SBATCH -t {args['time']}\n"
text += f"#SBATCH -J e3nn-MD_{name}\n\n"
text += f"python run_from_smiles.py '" + args["smiles"] + "'\n"
text += f"python hdf5_to_sp.py trj.hdf5 input/{name}-nmr\n"

with open("control.sh", "w+") as file:
    file.write(text)
print(f"{name}/control.sh created")

#### conda gets confused upon reloading ~/.bashrc so gotta toggle the environment
subprocess.call(['/bin/bash', '-i', '-c', "source deactivate; source activate openmm; sbatch control.sh"])

