import sys, h5py 
import numpy as np

path = sys.argv[1]
print(f"Reading from {path}")

gas_shifts, solv_shifts = None, None

with h5py.File(path, "r") as h5:
    gas_shifts = h5["gas_shifts"][:50000]
    solv_shifts = h5["solv_shifts"][tuple(map(slice, gas_shifts.shape))]

def stddev(X, block_size):
    mean_overall = np.mean(X)
    blocks = np.array_split(X, block_size)
    if len(blocks) == 1:
        return 0
    block_sq_mean = [np.mean(b) ** 2 for b in blocks]
    return np.sqrt(sum([b - mean_overall ** 2 for b in block_sq_mean]) / (len(blocks) - 1))

X = range(1,5000,1)
gY = [np.array(X)]
sY = [np.array(X)]

for n in range(len(gas_shifts[0])):
    print(f"Atom #{n}")
    print("gas phase:")
    print(f"Overall mean:\t{np.mean(gas_shifts[:,n]):.4f}")
    print(f"Overall stdev:\t{np.std(gas_shifts[:,n]):.4f}\n")
    print("solution phase:")
    print(f"Overall mean:\t{np.mean(solv_shifts[:,n]):.4f}")
    print(f"Overall stdev:\t{np.std(solv_shifts[:,n]):.4f}\n")
    gY.append(np.array([stddev(gas_shifts[:,0], m) for m in X]))
    sY.append(np.array([stddev(solv_shifts[:,0], m) for m in X]))


    np.savetxt("g_autocorr.txt", np.vstack(gY).T, delimiter=", ", fmt="%.3f")
    np.savetxt("s_autocorr.txt", np.vstack(sY).T, delimiter=", ", fmt="%.3f")
