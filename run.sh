#!/bin/bash

#  SBATCH CONFIG
#--------------------------------------------------------------------------------
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1                                        #nodes requested
#SBATCH --cpus-per-task=64                               #no. of cpu's per task
#SBATCH --ntasks=1                                       #no. of tasks (cpu cores)
#SBATCH --mem=32G                                        #memory requested
#SBATCH --job-name=factorio_model                         #job name
#SBATCH --time=12:00:00                                  #time limit in the form days-hours:minutes
#SBATCH --output=test_output%j.out                     #out file with unique jobid
#SBATCH --mail-type=BEGIN,FAIL,END                       #email sent to user during begin,fail and end of job
#SBATCH --mail-user=your_email@umsystem.edu              #<-- Make sure to use your email

echo "### Starting at: $(date) ###"

python dinov2.py