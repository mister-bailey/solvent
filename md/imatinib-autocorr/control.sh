#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --mem=34000
#SBATCH -t 7200
#SBATCH -J e3nn-MD_imatinib-autocorr

python run_from_smiles.py 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C'
python hdf5_to_sp.py trj.hdf5 input/imatinib-autocorr-nmr
