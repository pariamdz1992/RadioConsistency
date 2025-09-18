#!/bin/bash
#SBATCH --job-name=prepare_radiomapseer
#SBATCH --account=def-hinat
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=prepare_radiomapseer_%j.out
#SBATCH --error=prepare_radiomapseer_%j.err

# Load necessary modules
module load python/3.9
module load scipy-stack

# Activate your virtual environment
source ~/consistency_env/bin/activate

# Change to the datasets directory
cd /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/datasets

# Run the data preparation script
python prepare_data.py \
    --dataset_dir /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/data \
    --output_dir /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/datasets \
    --simulation DPM \
    --num-tx 80

echo "Data preparation completed successfully!"
