import numpy as np
import sys, os, shutil, argparse
import subprocess
import cctk

parser = argparse.ArgumentParser(prog="nmr_from_smiles.py")
parser.add_argument("--partition", "-p", type=str, default="serial_requeue")
parser.add_argument("--time", "-t", type=int, default=10080)
parser.add_argument("--cores", "-n", type=int, default=16)
parser.add_argument("--mem", "-m", type=int, default=2000) # per core, in MB
parser.add_argument("--force", "-F", default=False, action="store_true")
parser.add_argument("name", type=str)

args = vars(parser.parse_args(sys.argv[1:]))
name = args["name"]

molecule = cctk.Molecule.new_from_name(name)
print(f"generated molecule {name}")

if os.path.exists(name) and not args["force"]:
    os.chdir(name)
else:
    if os.path.exists(name):
        shutil.rmtree(name)

    os.mkdir(name)
    shutil.copyfile("default_run.config", f"{name}/run.config")
    shutil.copyfile("default_analyze.config", f"{name}/analyze.config")

    os.chdir(name)
    print(f"{name}/ created")
    os.mkdir("output")
    os.mkdir("analysis")

    cctk.GaussianFile.write_molecule_to_file(
        f"{name}.gjf",
        molecule,
        route_card="#p opt freq=(noraman, hpmodes)",
    )

    f = open("run.config",'r')
    filedata = f.read()
    f.close()
    newdata = filedata.replace("@NAME", name)
    f = open("run.config",'w')
    f.write(newdata)
    f.close()

    f = open("analyze.config",'r')
    filedata = f.read()
    f.close()
    newdata = filedata.replace("@NAME", name)
    f = open("analyze.config",'w')
    f.write(newdata)
    f.close()

#### create config file

text = "#!/bin/bash\n"
text += "#SBATCH -N 1\n"
text += f"#SBATCH -n {args['cores']}\n"
text += f"#SBATCH -p {args['partition']}\n"
text += f"#SBATCH --mem={args['cores']*args['mem'] + 2000}\n"
text += f"#SBATCH -t {args['time']}\n"
text += f"#SBATCH -J jprogdyn_{name}\n\n"
if not os.path.exists(f"output/{name}.out"):
    text += f"g16 {name}.gjf output/{name}.out\n"
text += "module load jdk/1.8.0_172-fasrc01\n"
text += "module load centos6/0.0.1-fasrc01\n"
text += "module load apache-maven/3.2.5-fasrc01\n"
text += "mvn exec:java -Dconfig.filename='run.config'\n"
text += "mvn exec:java -Dconfig.filename='analyze.config'\n"

with open("control.sh", "w+") as file:
    file.write(text)
print(f"{name}/control.sh created")

subprocess.call(['/bin/bash', '-i', '-c', "source deactivate; source activate openmm; sbatch control.sh"])
