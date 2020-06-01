import argparse, math, sys, os, subprocess, glob, re, shutil

# conda is being stupid and keeps running the wrong python
#os.environ["PATH"] =  os.path.expanduser("~") + "/.conda/envs/openmm/bin/python" + os.pathsep + os.environ["PATH"]

#### USAGE:
#### python submit.py "SMILES" name_of_molecule

#### OPTIONS
#### -p - partition name
#### -c - num cores
#### -t - time in minutes
#### -m - mem per core (MB)
#### -F - remove directory, if it already exists

#### Requires a copy of ``run_from_smiles.py`` and ``mdconf.yml`` in current directory.
#### Also, a conda environment named ``openmm``.
#### Corin Wagen, 2020

parser = argparse.ArgumentParser(prog="submit.py")
parser.add_argument("--partition", "-p", type=str, default="serial_requeue")
parser.add_argument("--time", "-t", type=int, default=7200)
parser.add_argument("--cores", "-c", type=int, default=16)
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
shutil.copyfile("mdconf.yml", f"{name}/mdconf.yml")
shutil.copyfile("run_from_smiles.py", f"{name}/run_from_smiles.py")

os.chdir(name)
print(f"{name}/ created")

text = "#!/bin/bash\n"
text += "#SBATCH -N 1\n"
text += f"#SBATCH -n {args['cores']}\n"
text += f"#SBATCH -p {args['partition']}\n"
text += f"#SBATCH --mem={args['cores']*args['mem'] + 2000}\n"
text += f"#SBATCH -t {args['time']}\n"
text += f"#SBATCH -J e3nn-MD_{name}\n\n"
text += f"python run_from_smiles.py '" + args["smiles"] + "'"

with open("control.sh", "w+") as file:
    file.write(text)
print(f"{name}/control.sh created")

#### conda gets confused upon reloading ~/.bashrc so gotta toggle the environment
subprocess.call(['/bin/bash', '-i', '-c', "source deactivate; source activate openmm; sbatch control.sh"])

