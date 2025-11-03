#!/usr/bin/env python3
"""
Test script_util_conditioned.py
Verifies that the new script utilities work correctly before proceeding to full training.
"""

import torch
import sys
from pathlib import Path

def setup_paths():
    """Setup paths for Compute Canada environment."""
    consistency_path = "/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models"
    if consistency_path not in sys.path:
        sys.path.insert(0, consistency_path)
    
    print(f"✓ Added path: {consistency_path}")
    return True

def test_script_util_imports():
    """Test that we can import the modified script utilities."""
    print("\n" + "="*60)
    print("TESTING SCRIPT_UTIL_CONDITIONED IMPORTS")
    print("="*60)
    
    try:
        # Test importing the conditioned script utils
        from cm.script_util_conditioned import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
            create_conditioned_model,
            cm_train_defaults,
        )
        print("✓ Successfully imported conditioned script utilities")
        
        # Test the defaults
        defaults = model_and_diffusion_defaults()
        print("✓ Retrieved model and diffusion defaults")
        
        # Check that new conditioning parameters are present
        conditioning_params = ['cond_net', 'cond_in_dim', 'fix_bb', 'in_channels', 'out_channels', 'use_conditioned_model']
        for param in conditioning_params:
            if param in defaults:
                print(f"✓ Found conditioning parameter: {param} = {defaults[param]}")
            else:
                print(f"✗ Missing conditioning parameter: {param}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("Make sure script_util_conditioned.py is in the cm/ directory")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_conditioned_model_creation():
    """Test creating the conditioned model through script utilities."""
    print("\n" + "="*60)
    print("TESTING CONDITIONED MODEL CREATION")
    print("="*60)
    
    try:
        from cm.script_util_conditioned import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
        )
        
        # Get default parameters
        defaults = model_and_diffusion_defaults()
        print("✓ Got default parameters")
        
        # Test creating conditioned model
        print("Creating conditioned model with defaults...")
        model, diffusion = create_model_and_diffusion(**defaults)
        
        print("✓ Successfully created conditioned model and diffusion")
        
        # Check model type
        from cm.conditioned_consistency_unet import ConditionedConsistencyUNet
        if isinstance(model, ConditionedConsistencyUNet):
            print("✓ Model is ConditionedConsistencyUNet (correct)")
        else:
            print(f"✗ Model is {type(model)} (should be ConditionedConsistencyUNet)")
            return False
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_compatibility():
    """Test that our new parameters work with the existing argument system."""
    print("\n" + "="*60)
    print("TESTING PARAMETER COMPATIBILITY")
    print("="*60)
    
    try:
        from cm.script_util_conditioned import (
            model_and_diffusion_defaults,
            cm_train_defaults,
            args_to_dict,
        )
        
        # Test that we can get all parameters
        model_defaults = model_and_diffusion_defaults()
        train_defaults = cm_train_defaults()
        
        print("✓ Retrieved all default parameters")
        
        # Test some specific radio parameters
        radio_params = {
            'image_size': 64,
            'cond_net': 'swin',
            'cond_in_dim': 3,
            'fix_bb': True,
            'in_channels': 1,
            'out_channels': 1,
            'use_conditioned_model': True,
        }
        
        # Test creating model with custom parameters
        combined_params = {**model_defaults, **radio_params}
        
        from cm.script_util_conditioned import create_model_and_diffusion
        model, diffusion = create_model_and_diffusion(**combined_params)
        
        print("✓ Successfully created model with custom radio parameters")
        print(f"  Image size: {radio_params['image_size']}")
        print(f"  Conditioning backbone: {radio_params['cond_net']}")
        print(f"  Conditioning channels: {radio_params['cond_in_dim']}")
        print(f"  Input channels: {radio_params['in_channels']}")
        print(f"  Output channels: {radio_params['out_channels']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test that the created model can actually process data."""
    print("\n" + "="*60)
    print("TESTING FORWARD PASS")
    print("="*60)
    
    try:
        from cm.script_util_conditioned import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
        )
        
        # Create model
        defaults = model_and_diffusion_defaults()
        model, diffusion = create_model_and_diffusion(**defaults)
        
        # Create test data matching your radio format
        batch_size = 2
        image_size = defaults['image_size']  # Should be 64
        
        radio_data = torch.randn(batch_size, 1, image_size, image_size)  # 1-channel radio maps
        conditioning = torch.randn(batch_size, 3, image_size, image_size)  # 3-channel conditioning
        timesteps = torch.randn(batch_size)
        
        print(f"✓ Created test data:")
        print(f"  Radio data: {radio_data.shape}")
        print(f"  Conditioning: {conditioning.shape}")
        print(f"  Timesteps: {timesteps.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(radio_data, timesteps, cond=conditioning)
        
        print("✓ Forward pass successful!")
        print(f"  Input: {radio_data.shape} → Output: {output.shape}")
        print(f"  Shape match: {output.shape == radio_data.shape}")
        
        # Check for NaN/Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("✗ NaN/Inf detected in output!")
            return False
        else:
            print("✓ Output values are valid (no NaN/Inf)")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("TESTING SCRIPT_UTIL_CONDITIONED.PY")
    print("Verifying conditioned script utilities work correctly")
    
    # Setup
    if not setup_paths():
        return False
    
    # Run tests
    tests = [
        ("Import Test", test_script_util_imports),
        ("Model Creation Test", test_conditioned_model_creation),
        ("Parameter Compatibility Test", test_parameter_compatibility),
        ("Forward Pass Test", test_forward_pass),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED ({passed}/{total})!")
        print("script_util_conditioned.py is working correctly!")
        print("\nReady to proceed to Step 2: Creating cm_train_conditioned.py")
    else:
        print(f"\n❌ SOME TESTS FAILED ({passed}/{total})")
        print("Fix the issues above before proceeding")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
