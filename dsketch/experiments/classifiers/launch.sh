#!/bin/bash -l
#SBATCH -p ecsstaff
#sbatch -u ecsstaff 
# SBATCH --mem=300G
# SBATCH --gres=gpu:4
# SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adm1g15@soton.ac.uk
#SBATCH --time=120:00:00

module load conda/py3-latest
conda activate sketching
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo $@
python /home/adm1g15/DifferentiableSketching/dsketch/experiments/classifiers/train.py $@