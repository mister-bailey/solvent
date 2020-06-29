#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p serial_requeue
#SBATCH --mem=34000
#SBATCH -t 60
#SBATCH -J e3nn-analysis

python analyze.py final_isotropic_shifts.hdf5
