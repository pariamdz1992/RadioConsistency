#!/usr/bin/env python3
"""
Test script for radio_image_datasets.py - Compute Canada Version
This version is adapted for cluster environments without interactive displays.
"""

import sys
import os
import torch
import numpy as np

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def test_radio_dataset_loader_cc():
    """Test the radio dataset loader on Compute Canada"""
    
    print("=" * 80)
    print("TESTING RADIO DATASET LOADER - COMPUTE CANADA VERSION")
    print("=" * 80)
    
    # =================================================================
    # Step 1: Environment check
    # =================================================================
    print("\n1. Checking environment...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Available files: {os.listdir('.')}")
    
    # =================================================================
    # Step 2: Test imports
    # =================================================================
    print("\n2. Testing imports...")
    try:
        from radio_datasets import RadioUNet_c, RadioUNet_s, load_data, RadioImageDataset
        print("✓ Successfully imported radio dataset classes")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("Make sure radio_image_datasets.py is in the current directory")
        return False
    
    # =================================================================
    # Step 3: Check for data directory
    # =================================================================
    print("\n3. Looking for data directory...")
    
    # Common Compute Canada data locations
    possible_data_dirs = [
        "/home/username/projects/def-supervisor/username/RadioMapSeer/",  # Project space
        "/home/username/scratch/RadioMapSeer/",  # Scratch space
        "/scratch/username/RadioMapSeer/",  # Alternative scratch
        "./RadioMapSeer/",  # Current directory
        "../RadioMapSeer/",  # Parent directory
        os.path.expanduser("~/RadioMapSeer/"),  # Home directory
    ]
    
    # Try to find data directory automatically
    data_dir = None
    for potential_dir in possible_data_dirs:
        if os.path.exists(potential_dir):
            data_dir = potential_dir
            print(f"✓ Found data directory: {data_dir}")
            break
    
    if data_dir is None:
        print("✗ Could not find RadioMapSeer data directory")
        print("Please update the script with your actual data path")
        print("Common locations on Compute Canada:")
        for pd in possible_data_dirs:
            print(f"  - {pd}")
        return False
    
    # Verify data structure
    expected_subdirs = ["gain", "png"]
    missing_dirs = []
    for subdir in expected_subdirs:
        full_path = os.path.join(data_dir, subdir)
        if not os.path.exists(full_path):
            missing_dirs.append(subdir)
    
    if missing_dirs:
        print(f"✗ Missing required subdirectories: {missing_dirs}")
        print(f"Data directory contents: {os.listdir(data_dir)}")
        return False
    
    print("✓ Data directory structure looks correct")
    
    # =================================================================
    # Step 4: Test RadioUNet_c creation
    # =================================================================
    print("\n4. Testing RadioUNet_c dataset creation...")
    try:
        radio_dataset = RadioUNet_c(
            phase="train",
            dir_dataset=data_dir,
            numTx=3,  # Small number for testing
            thresh=0.05,
            simulation="DPM",
            carsSimul="no",
            carsInput="no",
            cityMap="complete"
        )
        print(f"✓ Successfully created RadioUNet_c dataset with {len(radio_dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to create RadioUNet_c dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =================================================================
    # Step 5: Test loading a single sample
    # =================================================================
    print("\n5. Testing single sample loading...")
    try:
        sample = radio_dataset[0]
        print(f"✓ Successfully loaded sample")
        print(f"  Sample type: {type(sample)}")
        print(f"  Sample length: {len(sample)}")
        
        if len(sample) == 2:
            inputs, image_gain = sample
            print(f"  Inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
            print(f"  Image gain shape: {image_gain.shape}, dtype: {image_gain.dtype}")
            print(f"  Inputs range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            print(f"  Image gain range: [{image_gain.min():.3f}, {image_gain.max():.3f}]")
        else:
            print(f"  Unexpected sample format with {len(sample)} items")
            
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =================================================================
    # Step 6: Test wrapper without MPI (mock MPI for cluster)
    # =================================================================
    print("\n6. Testing RadioImageDataset wrapper...")
    try:
        # Mock MPI for cluster environment
        import unittest.mock
        with unittest.mock.patch('radio_image_datasets.MPI') as mock_mpi:
            mock_mpi.COMM_WORLD.Get_rank.return_value = 0
            mock_mpi.COMM_WORLD.Get_size.return_value = 1
            
            wrapper_dataset = RadioImageDataset(
                radio_dataset=radio_dataset,
                resolution=64,  # Small size for testing
                use_image_conditioning=True,
                classes=None,
                shard=0,
                num_shards=1
            )
            
            print(f"✓ Successfully created RadioImageDataset wrapper")
            print(f"  Wrapper length: {len(wrapper_dataset)}")
            
            # Test loading from wrapper
            target, conditioning_dict = wrapper_dataset[0]
            
            print(f"✓ Successfully loaded from wrapper")
            print(f"  Target shape: {target.shape}, dtype: {target.dtype}")
            print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")
            print(f"  Conditioning dict keys: {list(conditioning_dict.keys())}")
            
            if "conditioning" in conditioning_dict:
                cond = conditioning_dict["conditioning"]
                print(f"  Conditioning shape: {cond.shape}, dtype: {cond.dtype}")
                print(f"  Conditioning range: [{cond.min():.3f}, {cond.max():.3f}]")
            
    except Exception as e:
        print(f"✗ Failed wrapper test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =================================================================
    # Step 7: Test load_data function
    # =================================================================
    print("\n7. Testing load_data function...")
    try:
        with unittest.mock.patch('radio_image_datasets.MPI') as mock_mpi:
            mock_mpi.COMM_WORLD.Get_rank.return_value = 0
            mock_mpi.COMM_WORLD.Get_size.return_value = 1
            
            data_generator = load_data(
                data_dir=data_dir,
                batch_size=2,
                image_size=64,
                class_cond=False,
                use_image_conditioning=True,
                dataset_type="RadioUNet_c",
                simulation="DPM",
                carsSimul="no",
                carsInput="no",
                cityMap="complete",
                numTx=2  # Small number for testing
            )
            
            # Get one batch
            batch = next(data_generator)
            images, kwargs = batch
            
            print(f"✓ Successfully loaded batch from load_data")
            print(f"  Batch images shape: {images.shape}")
            print(f"  Batch images range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  Batch kwargs keys: {list(kwargs.keys())}")
            
            if "conditioning" in kwargs:
                cond_batch = kwargs["conditioning"]
                print(f"  Batch conditioning shape: {cond_batch.shape}")
                print(f"  Batch conditioning range: [{cond_batch.min():.3f}, {cond_batch.max():.3f}]")
            
    except Exception as e:
        print(f"✗ Failed load_data test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =================================================================
    # Step 8: Create visualization (save to file)
    # =================================================================
    print("\n8. Creating visualization...")
    try:
        target, cond_dict = wrapper_dataset[0]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Show target (radio gain map)
        target_img = (target[0] + 1) / 2  # Convert from [-1,1] to [0,1]
        axes[0].imshow(target_img, cmap='hot')
        axes[0].set_title('Target (Radio Gain)')
        axes[0].axis('off')
        
        # Show conditioning channels
        if "conditioning" in cond_dict:
            cond = cond_dict["conditioning"]
            
            for i in range(min(3, cond.shape[0])):
                cond_img = (cond[i] + 1) / 2  # Convert from [-1,1] to [0,1]
                axes[i+1].imshow(cond_img, cmap='gray')
                
                if i == 0:
                    axes[i+1].set_title('Buildings')
                elif i == 1:
                    axes[i+1].set_title('Transmitters')
                elif i == 2:
                    axes[i+1].set_title('Cars/Other')
                    
                axes[i+1].axis('off')
        
        plt.tight_layout()
        
        # Save to results directory if it exists, otherwise current directory
        if os.path.exists('./results'):
            save_path = './results/radio_dataset_test_visualization.png'
        else:
            save_path = './radio_dataset_test_visualization.png'
            
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")
        print("  This is not critical for the dataset loader functionality")
    
    # =================================================================
    # Step 9: Performance test
    # =================================================================
    print("\n9. Performance test...")
    try:
        import time
        
        # Time loading multiple batches
        start_time = time.time()
        num_batches = 5
        
        with unittest.mock.patch('radio_image_datasets.MPI') as mock_mpi:
            mock_mpi.COMM_WORLD.Get_rank.return_value = 0
            mock_mpi.COMM_WORLD.Get_size.return_value = 1
            
            data_generator = load_data(
                data_dir=data_dir,
                batch_size=4,
                image_size=64,
                use_image_conditioning=True,
                dataset_type="RadioUNet_c",
                numTx=5
            )
            
            for i in range(num_batches):
                batch = next(data_generator)
                
        end_time = time.time()
        total_time = end_time - start_time
        samples_per_second = (num_batches * 4) / total_time
        
        print(f"✓ Performance test completed")
        print(f"  Loaded {num_batches} batches (4 samples each) in {total_time:.2f} seconds")
        print(f"  Performance: {samples_per_second:.1f} samples/second")
        
    except Exception as e:
        print(f"⚠ Performance test failed: {e}")
    
    # =================================================================
    # Success!
    # =================================================================
    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED ON COMPUTE CANADA! 🎉")
    print("Your radio dataset loader is working correctly!")
    print("=" * 80)
    
    print(f"\nTested with data from: {data_dir}")
    print("\nNext steps:")
    print("1. Your dataset loader is ready for consistency model training")
    print("2. Submit a job to train the consistency model")
    print("3. Use the load_data function in your training script")
    
    return True


if __name__ == "__main__":
    print("Radio Dataset Loader Test - Compute Canada Version")
    print("This test is designed for cluster environments")
    print()
    
    success = test_radio_dataset_loader_cc()
    
    if success:
        print("\n✅ Ready for training on Compute Canada!")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
        sys.exit(1)
