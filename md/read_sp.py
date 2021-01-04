import h5py, cctk, logging, glob, yaml, sys, tqdm, re
import numpy as np
import multiprocessing as mp

#### USAGE: python read_sp.py "path/to/output/*.out"

logging.basicConfig(level=logging.INFO, filename="log.log", format='%(asctime)s %(name)-12s  %(message)s', datefmt='%m-%d %H:%M', filemode="a")
logger = logging.getLogger(__name__)

filenames = glob.glob(sys.argv[1], recursive=True)
logger.info(f"Reading {len(filenames)} filenames from {sys.argv[1]}")

config = yaml.load(open("config.yml", "r"), yaml.Loader)
num_files = config["num_nmr_files"]

gas_num = 33 
solv_num = 83
max_time = 1000000
hdf5_path = "final_isotropic_shifts.hdf5"

def read(filename):
    shifts = []
    times = []
    files = []
    count = 0

    try:
        files = cctk.GaussianFile.read_file(filename)
        assert isinstance(files, list)

        for f in files:
            if f.successful_terminations:
                shifts.append(f.ensemble[-1, "isotropic_shielding"])
                match = re.search("MD frame from t=(.+?) picoseconds", f.title)
                times.append(int(float(match.group(1)) * 1000 + 0.5))
                count += 1
            else: 
                logger.info(f"{filename} contained failed job!")
    except Exception as e:
        print(f"skipping {filename}!\n{e}")
  
    return [shifts, times, count]
        
final_gas_shifts = np.zeros(shape=(max_time, gas_num))
final_solv_shifts = np.zeros(shape=(max_time, solv_num))

pool = mp.Pool(processes=config["num_threads"])
count = 0
for i, output in enumerate(tqdm.tqdm(pool.imap(read, filenames), total=len(filenames))):
    count += output[2]
    for shifts, time in zip(output[0], output[1]):
        if shifts is None:
            continue

        if len(shifts) == gas_num:
            final_gas_shifts[time] = shifts
        elif len(shifts) == solv_num:
            final_solv_shifts[time] = shifts
        else:
            logger.info("Wrong number of shifts! {len(shifts)} doesn't match either gas or solution.")

with h5py.File(hdf5_path, "w") as h5: 
    h5.create_dataset("gas_shifts", data=final_gas_shifts)
    h5.create_dataset("solv_shifts", data=final_solv_shifts)

print("final count:")
print(count/2)
logger.info(f"Wrote data to {hdf5_path}")
