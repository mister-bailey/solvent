#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -p serial_requeue
#SBATCH --mem=34000
#SBATCH -t 7200
#SBATCH -J e3nn-MD_acrolein

python run_from_smiles.py 'C=CC=O'