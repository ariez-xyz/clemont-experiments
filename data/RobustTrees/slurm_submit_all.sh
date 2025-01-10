#!/bin/bash
#
# To submit: sbatch script.sh
#
#SBATCH --partition=defaultp
#
#
#
#-------------------------------------------------------------
#example script for running a single-CPU serial job via SLURM
#-------------------------------------------------------------
#
#SBATCH --job-name=robusttrees-inference
#SBATCH --output=robusttrees-inference--%j.log   
#            %j is a placeholder for the jobid
#
#Define the number of hours the job should run. 
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=32:00:00
#
# Define the amount of RAM used by your job in GigaBytes
#SBATCH --mem=16G
# Use 8 cores
#SBATCH -c 8

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # Not sure if needed?

#Pick whether you prefer requeue or not. If you use the --requeue
#option, the requeued job script will start from the beginning, 
#potentially overwriting your previous progress, so be careful.
#For some people the --requeue option might be desired if their
#application will continue from the last state.
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#
#Do not export the local environment to the compute nodes
#SBATCH --export=ALL

#
#load the respective software module you intend to use
#
#
#run the respective binary through SLURM's srun

pushd ../..
source activate.sh
popd

./infer_binary_mnist.sh # Broken? Won't train past checkpoint 0009
./infer_breast_cancer.sh
./infer_cod_rna.sh
./infer_covtype.sh
./infer_diabetes.sh
./infer_fashion_mnist.sh
./infer_ijcnn.sh
./infer_sensorless.sh
./infer_webspam.sh
./infer_mnist.sh 
./infer_higgs.sh # Takes longest

