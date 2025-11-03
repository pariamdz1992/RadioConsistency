#!/usr/bin/env python3
"""
Compute Canada Verification Script for Conditioned Consistency U-Net
Tests the exact conditioning architecture with radio map data setup.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

def setup_paths():
    """Setup paths for Compute Canada environment."""
    # Add your consistency models path - using your exact path structure
    consistency_path = "/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models"
    if consistency_path not in sys.path:
        sys.path.insert(0, consistency_path)
    
    print(f"✓ Added path: {consistency_path}")
    
    # Verify the path exists
    cm_path = Path(consistency_path) / "cm"
    if not cm_path.exists():
        print(f"✗ Error: {cm_path} does not exist!")
        print("Make sure you've copied the files to the correct location.")
        print(f"Expected: {cm_path}")
        return False
    
    return True

def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    try:
        from cm.conditioned_consistency_unet import ConditionedConsistencyUNet
        print("✓ Successfully imported ConditionedConsistencyUNet")
    except ImportError as e:
        print(f"✗ Failed to import ConditionedConsistencyUNet: {e}")
        print("\nCheck that you have:")
        print("1. Copied conditioned_consistency_unet.py to cm/")
        print("2. Copied backbone files (swin_transformer.py, etc.) to cm/")
        return False
    
    # Test backbone imports
    backbone_files = [
        "swin_transformer", "efficientnet", "resnet", "vgg"
    ]
    
    for backbone in backbone_files:
        try:
            module = __import__(f"cm.{backbone}", fromlist=[backbone])
            print(f"✓ Successfully imported {backbone}")
        except ImportError as e:
            print(f"✗ Failed to import {backbone}: {e}")
            print(f"  Make sure {backbone}.py is copied to cm/")
            return False
    
    return True

def test_model_creation():
    """Test creating the conditioned model with different backbones."""
    print("\n" + "="*60)
    print("TESTING MODEL CREATION")
    print("="*60)
    
    from cm.conditioned_consistency_unet import ConditionedConsistencyUNet
    
    # Your radio data parameters
    radio_params = {
        'dim': 128,
        'dim_mults': (1, 2, 4, 4),
        'channels': 1,
        'cond_in_dim': 3,  # Buildings, transmitters, cars
        'fix_bb': True,
        'window_sizes1': [[8, 8], [4, 4], [2, 2], [1, 1]],
        'window_sizes2': [[8, 8], [4, 4], [2, 2], [1, 1]],
        'input_size': [256, 256]  # Adjust to your radio map size
    }
    
    # Test each backbone type
    backbone_types = ['swin', 'effnet', 'resnet', 'vgg']
    successful_backbones = []
    
    for cond_net in backbone_types:
        print(f"\n--- Testing {cond_net.upper()} backbone ---")
        
        try:
            model = ConditionedConsistencyUNet(
                cond_net=cond_net,
                **radio_params
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"✓ {cond_net} model created successfully")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Frozen parameters: {total_params - trainable_params:,}")
            
            successful_backbones.append(cond_net)
            
        except Exception as e:
            print(f"✗ {cond_net} model creation failed: {e}")
    
    return successful_backbones

def test_forward_pass(successful_backbones):
    """Test forward pass with simulated radio data."""
    print("\n" + "="*60)
    print("TESTING FORWARD PASS")
    print("="*60)
    
    if not successful_backbones:
        print("✗ No successful backbones to test forward pass")
        return False
    
    from cm.conditioned_consistency_unet import ConditionedConsistencyUNet
    
    # Simulate your radio data dimensions
    batch_size = 2
    radio_height, radio_width = 256, 256  # Adjust to your actual size
    
    # Create test data (simulating your radio dataset format)
    print(f"Creating test data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Radio maps: {batch_size} x 1 x {radio_height} x {radio_width}")
    print(f"  Conditioning: {batch_size} x 3 x {radio_height} x {radio_width}")
    
    radio_maps = torch.randn(batch_size, 1, radio_height, radio_width)
    conditioning = torch.randn(batch_size, 3, radio_height, radio_width)  # Buildings, transmitters, cars
    timesteps = torch.randn(batch_size)
    
    # Test with the first successful backbone
    test_backbone = successful_backbones[0]
    print(f"\nTesting forward pass with {test_backbone} backbone...")
    
    try:
        model = ConditionedConsistencyUNet(
            dim=128,
            dim_mults=(1, 2, 4, 4),
            channels=1,
            cond_in_dim=3,
            cond_net=test_backbone,
            fix_bb=True,
            input_size=[radio_height, radio_width]
        )
        
        # Set to evaluation mode for testing
        model.eval()
        
        with torch.no_grad():
            print("  Running forward pass...")
            output = model(radio_maps, timesteps, cond=conditioning)
            
            print("✓ Forward pass successful!")
            print(f"  Input shape: {radio_maps.shape}")
            print(f"  Conditioning shape: {conditioning.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Shape consistency: {output.shape == radio_maps.shape}")
            
            # Check output is reasonable
            output_mean = output.mean().item()
            output_std = output.std().item()
            print(f"  Output statistics: mean={output_mean:.4f}, std={output_std:.4f}")
            
            return True
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conditioning_requirement():
    """Test that conditioning is properly required."""
    print("\n" + "="*60)
    print("TESTING CONDITIONING REQUIREMENT")
    print("="*60)
    
    from cm.conditioned_consistency_unet import ConditionedConsistencyUNet
    
    try:
        model = ConditionedConsistencyUNet(
            dim=128,
            dim_mults=(1, 2, 4, 4),
            channels=1,
            cond_in_dim=3,
            cond_net='swin'
        )
        
        # Test without conditioning (should fail)
        radio_maps = torch.randn(1, 1, 256, 256)
        timesteps = torch.randn(1)
        
        try:
            with torch.no_grad():
                output = model(radio_maps, timesteps)  # No conditioning
            print("✗ Model should require conditioning!")
            return False
        except ValueError as e:
            print("✓ Model properly requires conditioning")
            print(f"  Error message: {e}")
            return True
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability on Compute Canada."""
    print("\n" + "="*60)
    print("GPU AVAILABILITY CHECK")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"✓ CUDA available")
        print(f"  GPU count: {gpu_count}")
        print(f"  Current device: {current_device}")
        print(f"  GPU name: {gpu_name}")
        
        # Test GPU memory
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f"✓ GPU memory test passed")
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"  GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU memory test failed: {e}")
            
    else:
        print("✗ CUDA not available - running on CPU")
        print("  This is fine for testing, but you'll want GPU for training")

def main():
    """Main verification function."""
    print("="*60)
    print("CONDITIONED CONSISTENCY U-NET VERIFICATION")
    print("Compute Canada Environment")
    print("="*60)
    
    # Setup paths
    if not setup_paths():
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    # Test model creation
    successful_backbones = test_model_creation()
    if not successful_backbones:
        print("\n✗ VERIFICATION FAILED: No backbones working")
        return False
    
    # Test forward pass
    if not test_forward_pass(successful_backbones):
        print("\n✗ VERIFICATION FAILED: Forward pass failed")
        return False
    
    # Test conditioning requirement
    if not test_conditioning_requirement():
        print("\n✗ VERIFICATION FAILED: Conditioning not properly required")
        return False
    
    # Check GPU
    check_gpu_availability()
    
    # Success summary
    print("\n" + "="*60)
    print("🎉 VERIFICATION SUCCESSFUL! 🎉")
    print("="*60)
    print("Your conditioned consistency U-Net is working correctly!")
    print(f"\nSuccessful backbones: {', '.join(successful_backbones)}")
    print("\nNext steps:")
    print("1. Update your training script to use ConditionedConsistencyUNet")
    print("2. Add conditioning parameters to your training arguments")
    print("3. Test with a small subset of your radio dataset")
    print("4. Start full training!")
    print(f"\nRecommended backbone for your radio data: swin")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
