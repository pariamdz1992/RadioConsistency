import sys
import torch
import numpy as np
from mpi4py import MPI

# Import the radio dataset functions
from cm.radio_image_datasets import load_radio_data, RadioDataset

def test_radio_dataset():
    """Test the RadioDataset class directly"""
    print("=" * 50)
    print("Testing RadioDataset class...")
    
    try:
        # Create dataset instance
        dataset = RadioDataset(
            data_dir="/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/data/",
            image_size=256,
            dataset_variant="radio",
            phase="train",
            shard=0,
            num_shards=1,
        )
        
        print(f"Dataset created successfully!")
        print(f"Dataset length: {len(dataset)}")
        
        # Test getting one sample
        if len(dataset) > 0:
            print("\nTesting __getitem__...")
            image_gain, out_dict = dataset[0]
            
            print(f"Image shape: {image_gain.shape}")
            print(f"Image dtype: {image_gain.dtype}")
            print(f"Image min/max: {image_gain.min():.3f} / {image_gain.max():.3f}")
            
            print(f"Out_dict keys: {out_dict.keys()}")
            if 'cond' in out_dict:
                cond = out_dict['cond']
                print(f"Condition shape: {cond.shape}")
                print(f"Condition dtype: {cond.dtype}")
                print(f"Condition min/max: {cond.min():.3f} / {cond.max():.3f}")
        
        print("✅ RadioDataset test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ RadioDataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_load_radio_data():
    """Test the load_radio_data function"""
    print("=" * 50)  
    print("Testing load_radio_data function...")
    
    try:
        # Create data loader
        data_loader = load_radio_data(
            data_dir="/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/data/",
            batch_size=2,
            image_size=256,
            dataset_variant="radio",
            phase="train",
            deterministic=True,
        )
        
        print("Data loader created successfully!")
        
        # Get one batch
        print("\nGetting first batch...")
        images, model_kwargs = next(data_loader)
        
        print(f"Batch images shape: {images.shape}")
        print(f"Batch images dtype: {images.dtype}")
        print(f"Batch images min/max: {images.min():.3f} / {images.max():.3f}")
        
        print(f"Model kwargs keys: {model_kwargs.keys()}")
        if 'cond' in model_kwargs:
            cond = model_kwargs['cond']
            print(f"Batch conditions shape: {cond.shape}")
            print(f"Batch conditions dtype: {cond.dtype}")
            print(f"Batch conditions min/max: {cond.min():.3f} / {cond.max():.3f}")
        
        print("✅ load_radio_data test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ load_radio_data test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_format():
    """Test that data format matches consistency model expectations"""
    print("=" * 50)
    print("Testing data format compatibility...")
    
    try:
        data_loader = load_radio_data(
            data_dir="/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/data/",
            batch_size=1,
            image_size=256,
            dataset_variant="radio",
            phase="train",
            deterministic=True,
        )
        
        images, model_kwargs = next(data_loader)
        
        # Check image format (should be NCHW, [-1,1] range)
        assert len(images.shape) == 4, f"Expected 4D tensor, got {images.shape}"
        assert images.shape[1] == 1, f"Expected 1 channel (grayscale), got {images.shape[1]}"
        assert images.shape[2] == images.shape[3] == 256, f"Expected 256x256, got {images.shape[2:4]}"
        
        # Check value range (should be approximately [-1, 1])
        assert images.min() >= -1.1, f"Images too negative: {images.min()}"
        assert images.max() <= 1.1, f"Images too positive: {images.max()}"
        
        # Check conditioning format
        assert 'cond' in model_kwargs, "Missing 'cond' key in model_kwargs"
        cond = model_kwargs['cond']
        assert len(cond.shape) == 4, f"Expected 4D condition tensor, got {cond.shape}"
        assert cond.shape[1] == 2, f"Expected 2 channels (buildings, Tx), got {cond.shape[1]}"
        
        print("✅ Data format test PASSED!")
        print(f"  - Images: {images.shape} in range [{images.min():.3f}, {images.max():.3f}]")
        print(f"  - Conditions: {cond.shape} in range [{cond.min():.3f}, {cond.max():.3f}]")
        return True
        
    except Exception as e:
        print(f"❌ Data format test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Starting Radio Dataset Tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPI rank: {MPI.COMM_WORLD.Get_rank()}")
    print(f"MPI size: {MPI.COMM_WORLD.Get_size()}")
    
    # Run tests
    tests = [
        test_radio_dataset,
        test_load_radio_data,
        test_data_format,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
