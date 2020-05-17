import numpy as np
import h5py
import pandas as pd
from pandas import DataFrame, Series

# read equilibrium geometry data
with h5py.File("acetone-b3lyp_d3bj-631gd-gas-equilibrium_geometry.hdf5", "r") as h5:
    equilibrium_geometry = np.array(h5.get("equilibrium_geometry"))
    atomic_numbers = np.array(h5.get("atomic_numbers"))
    isotropic_shieldings = np.array(h5.get("isotropic_shieldings"))
    hirshfeld_charges = np.array(h5.get("hirshfeld_charges"))
    mulliken_charges = np.array(h5.get("mulliken_charges"))

# print out equilibrium data
print("equilibrium geometry")
print(equilibrium_geometry)
equilibrium_list = [atomic_numbers, isotropic_shieldings, hirshfeld_charges, mulliken_charges]
equilibrium_list = np.array(equilibrium_list).T.copy()
columns = ["Z", "sigma", "z_hir", "z_mul"]
equilibrium_df = DataFrame(equilibrium_list, columns=columns)
equilibrium_df.index = [ i+1 for i in range(len(equilibrium_df)) ]
equilibrium_df.index.name = "atom#"
print()
print("equilibrium properties")
print(equilibrium_df)

# read near-equilbrium data
with h5py.File("acetone-b3lyp_d3bj-631gd-gas-NMR-pcSseg_1.hdf5", "r") as h5:
    geoms_and_shieldings = np.array(h5.get("data"))

# print out non-equilbrium data
print("near-equilibrium data")
shape = np.shape(geoms_and_shieldings)
print(f"there are {shape[0]} geometries ({shape[1]} atoms per geometry)")
print()
geometries = geoms_and_shieldings[:,:,:3]
shieldings = geoms_and_shieldings[:,:,-1]
geoms_and_shieldings = None
print("first geometry")
print(geometries[0])
print()
print("first set of shieldings")
print(shieldings[0])

