#!/bin/bash
#SBATCH --job-name=radio_test_fixed
#SBATCH --account=def-hinat
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.11 scipy-stack cuda/12.2 mpi4py opencv
source ~/consistency_models_env/bin/activate
cd /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models
export CUDA_VISIBLE_DEVICES=0

python -c "
print('=' * 80)
print('RADIO DATASET LOADER TEST - FIXED PATHS')
print('=' * 80)

import sys
import os
import torch
import numpy as np

# Set matplotlib backend for headless environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =================================================================
# Step 1: Environment check
# =================================================================
print('\\n1. Checking environment...')
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
print(f'Current working directory: {os.getcwd()}')

# =================================================================
# Step 2: Check file location and import
# =================================================================
print('\\n2. Checking file location and testing imports...')

# Check if the file exists at the expected location
radio_datasets_path = '/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/cm/radio_datasets.py'
if os.path.exists(radio_datasets_path):
    print(f'✓ Found radio_datasets.py at: {radio_datasets_path}')
else:
    print(f'✗ File not found at: {radio_datasets_path}')
    # Check what's in the cm directory
    cm_dir = '/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/cm/'
    if os.path.exists(cm_dir):
        print(f'Contents of cm directory: {os.listdir(cm_dir)}')
    exit(1)

# Add the cm directory to Python path
sys.path.insert(0, '/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/cm')

try:
    from radio_datasets import RadioUNet_c, RadioUNet_s, load_data, RadioImageDataset
    print('✓ Successfully imported radio dataset classes')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    exit(1)

# =================================================================
# Step 3: Check data directory
# =================================================================
print('\\n3. Checking data directory...')
data_dir = '/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/data/'

if not os.path.exists(data_dir):
    print(f'✗ Data directory not found: {data_dir}')
    exit(1)

print(f'✓ Found data directory: {data_dir}')
print(f'Data directory contents: {os.listdir(data_dir)}')

# Check for RadioMapSeer subdirectory or use data_dir directly
if 'RadioMapSeer' in os.listdir(data_dir):
    data_dir = os.path.join(data_dir, 'RadioMapSeer')
    print(f'✓ Using RadioMapSeer subdirectory: {data_dir}')
elif 'gain' in os.listdir(data_dir) and 'png' in os.listdir(data_dir):
    print(f'✓ Using data directory directly: {data_dir}')
else:
    print(f'✗ Expected structure not found in: {data_dir}')
    print(f'Looking for \\'gain\\' and \\'png\\' subdirectories')
    exit(1)

# =================================================================
# Step 4: Test RadioUNet_c creation
# =================================================================
print('\\n4. Testing RadioUNet_c dataset creation...')
try:
    radio_dataset = RadioUNet_c(
        phase='train',
        dir_dataset=data_dir,
        numTx=3,  # Small number for testing
        thresh=0.05,
        simulation='DPM',
        carsSimul='no',
        carsInput='no',
        cityMap='complete'
    )
    print(f'✓ Successfully created RadioUNet_c dataset with {len(radio_dataset)} samples')
except Exception as e:
    print(f'✗ Failed to create RadioUNet_c dataset: {e}')
    import traceback
    traceback.print_exc()
    exit(1)

# =================================================================
# Step 5: Test loading a single sample
# =================================================================
print('\\n5. Testing single sample loading...')
try:
    sample = radio_dataset[0]
    print(f'✓ Successfully loaded sample')
    print(f'  Sample type: {type(sample)}')
    print(f'  Sample length: {len(sample)}')
    
    if len(sample) == 2:
        inputs, image_gain = sample
        print(f'  Inputs shape: {inputs.shape}, dtype: {inputs.dtype}')
        print(f'  Image gain shape: {image_gain.shape}, dtype: {image_gain.dtype}')
        print(f'  Inputs range: [{inputs.min():.3f}, {inputs.max():.3f}]')
        print(f'  Image gain range: [{image_gain.min():.3f}, {image_gain.max():.3f}]')
    
except Exception as e:
    print(f'✗ Failed to load sample: {e}')
    import traceback
    traceback.print_exc()
    exit(1)

# =================================================================
# Step 6: Test wrapper with MPI
# =================================================================
print('\\n6. Testing RadioImageDataset wrapper...')
try:
    from mpi4py import MPI
    
    wrapper_dataset = RadioImageDataset(
        radio_dataset=radio_dataset,
        resolution=64,  # Small size for testing
        use_image_conditioning=True,
        classes=None,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    
    print(f'✓ Successfully created RadioImageDataset wrapper')
    print(f'  MPI rank: {MPI.COMM_WORLD.Get_rank()}')
    print(f'  MPI size: {MPI.COMM_WORLD.Get_size()}')
    print(f'  Wrapper length: {len(wrapper_dataset)}')
    
    # Test loading from wrapper
    target, conditioning_dict = wrapper_dataset[0]
    
    print(f'✓ Successfully loaded from wrapper')
    print(f'  Target shape: {target.shape}, dtype: {target.dtype}')
    print(f'  Target range: [{target.min():.3f}, {target.max():.3f}]')
    print(f'  Conditioning dict keys: {list(conditioning_dict.keys())}')
    
    if 'conditioning' in conditioning_dict:
        cond = conditioning_dict['conditioning']
        print(f'  Conditioning shape: {cond.shape}, dtype: {cond.dtype}')
        print(f'  Conditioning range: [{cond.min():.3f}, {cond.max():.3f}]')
    
except Exception as e:
    print(f'✗ Failed wrapper test: {e}')
    import traceback
    traceback.print_exc()
    exit(1)

# =================================================================
# Step 7: Test load_data function
# =================================================================
print('\\n7. Testing load_data function...')
try:
    data_generator = load_data(
        data_dir=data_dir,
        batch_size=2,
        image_size=64,
        class_cond=False,
        use_image_conditioning=True,
        dataset_type='RadioUNet_c',
        simulation='DPM',
        carsSimul='no',
        carsInput='no',
        cityMap='complete',
        numTx=2  # Small number for testing
    )
    
    # Get one batch
    batch = next(data_generator)
    images, kwargs = batch
    
    print(f'✓ Successfully loaded batch from load_data')
    print(f'  Batch images shape: {images.shape}')
    print(f'  Batch images range: [{images.min():.3f}, {images.max():.3f}]')
    print(f'  Batch kwargs keys: {list(kwargs.keys())}')
    
    if 'conditioning' in kwargs:
        cond_batch = kwargs['conditioning']
        print(f'  Batch conditioning shape: {cond_batch.shape}')
        print(f'  Batch conditioning range: [{cond_batch.min():.3f}, {cond_batch.max():.3f}]')
    
    # Test GPU transfer
    if torch.cuda.is_available():
        images_gpu = images.cuda()
        print(f'✓ Successfully moved batch to GPU')
        print(f'  GPU batch shape: {images_gpu.shape}')
        print(f'  GPU batch device: {images_gpu.device}')
    
except Exception as e:
    print(f'✗ Failed load_data test: {e}')
    import traceback
    traceback.print_exc()
    exit(1)

# =================================================================
# Step 8: Test performance
# =================================================================
print('\\n8. Performance test...')
try:
    import time
    
    start_time = time.time()
    num_batches = 3
    
    perf_data_gen = load_data(
        data_dir=data_dir,
        batch_size=4,
        image_size=64,
        use_image_conditioning=True,
        dataset_type='RadioUNet_c',
        numTx=3
    )
    
    for i in range(num_batches):
        batch = next(perf_data_gen)
        
    end_time = time.time()
    total_time = end_time - start_time
    samples_per_second = (num_batches * 4) / total_time
    
    print(f'✓ Performance test completed')
    print(f'  Loaded {num_batches} batches (4 samples each) in {total_time:.2f} seconds')
    print(f'  Performance: {samples_per_second:.1f} samples/second')
    
except Exception as e:
    print(f'⚠ Performance test failed: {e}')

# =================================================================
# Success!
# =================================================================
print('\\n' + '=' * 80)
print('🎉 ALL TESTS PASSED! 🎉')
print('Your radio dataset loader is working correctly!')
print('=' * 80)

print(f'\\nTested with data from: {data_dir}')
print('\\nNext steps:')
print('1. Update your training script to import from cm.radio_datasets')
print('2. Use the load_data function in your consistency model training')

print('\\nExample import in training script:')
print('from cm.radio_datasets import load_data')

print('\\nExample training command:')
print('python cm_train.py \\\\')
print(f'    --data_dir {data_dir} \\\\')
print('    --use_image_conditioning True \\\\')
print('    --dataset_type RadioUNet_c \\\\')
print('    --simulation DPM \\\\')
print('    --image_size 256 \\\\')
print('    --batch_size 32')
"
