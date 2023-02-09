#!/bin/bash
#SBATCH -J getS
#SBATCH -n 1             # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 7-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared,unrestricted          # Partition to submit to
#SBATCH --mem 5gb           # Memory pool for all cores (see also --me$
#SBATCH -o getS_%j.out  # File to which STDOUT will be written, %j i$
#SBATCH -e getS_%j.err  # File to which STDERR will be written, %j i$
#SBATCH --mail-type=ALL      # Email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=txiong@g.harvard.edu

source /n/home00/txiong/miniconda3/etc/profile.d/conda.sh
conda activate EnvPy3.9

cd /n/home00/txiong/Research/2022_EntropyAncestry/Simulation_SLiM/NeutralWF

dir=$1
seed=$2
P=$3
nDiploid=$4
L=$5
nResampling=$6

echo Command line:
echo python3 slim_get_entropy.py --dir ${dir} --seed ${seed} --P ${P} --nDiploid ${nDiploid} --L ${L} --nResampling ${nResampling}

python3 slim_get_entropy.py --dir ${dir} --seed ${seed} --P ${P} --nDiploid ${nDiploid} --L ${L} --nResampling ${nResampling}