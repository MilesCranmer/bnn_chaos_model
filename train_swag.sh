#!/bin/bash
#SBATCH -N1
#SBATCH --job-name=var_reg
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100000
#SBATCH -p gpu
#SBATCH --constraint="v100|p100"

run_string="/mnt/home/mcranmer/miniconda3/envs/main2/bin/python swag_part1.py $@ || /mnt/home/mcranmer/miniconda3/envs/main2/bin/python swag_part2.py $@"
#run_string="/mnt/home/mcranmer/miniconda3/envs/main2/bin/python swag_part2.py $@"

srun bash -c "$run_string"
