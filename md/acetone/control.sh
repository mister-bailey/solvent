#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --mem=34000
#SBATCH -t 7200
#SBATCH -J e3nn-MD_acetone

python run_from_smiles.py 'CC(=O)C'