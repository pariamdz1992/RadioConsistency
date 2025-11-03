#!/usr/bin/env python3
"""
Radio Data Explorer - Understand your exact data format
This script examines your radio data directory structure and formats.
"""

import torch
import sys
import os
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image
import json

def setup_paths():
    """Setup paths for Compute Canada environment."""
    consistency_path = "/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models"
    if consistency_path not in sys.path:
        sys.path.insert(0, consistency_path)
    
    data_path = "/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/data"
    if data_path not in sys.path:
        sys.path.insert(0, data_path)
    
    return consistency_path, data_path

def explore_data_structure(data_dir):
    """Explore the structure of your radio data directory."""
    print("="*60)
    print("EXPLORING RADIO DATA STRUCTURE")
    print("="*60)
    
    data_path = Path(data_dir)
    print(f"Data directory: {data_path}")
    
    # Examine each subdirectory
    subdirs = ['antenna', 'gain', 'png', 'polygon']
    
    for subdir in subdirs:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            print(f"\n📁 {subdir}/ directory:")
            files = list(subdir_path.iterdir())
            print(f"   {len(files)} total items")
            
            # Show first few files
            for i, file in enumerate(files[:5]):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024*1024)
                    print(f"   📄 {file.name} ({size_mb:.2f} MB)")
                elif file.is_dir():
                    subfiles = len(list(file.glob('*')))
                    print(f"   📁 {file.name}/ ({subfiles} files)")
            
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
        else:
            print(f"\n❌ {subdir}/ directory not found")
    
    # Examine dataset.csv
    csv_path = data_path / "dataset.csv"
    if csv_path.exists():
        print(f"\n📄 dataset.csv:")
        try:
            df = pd.read_csv(csv_path)
            print(f"   {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {list(df.columns)}")
            
            if len(df) > 0:
                print(f"   First 3 rows:")
                for i, row in df.head(3).iterrows():
                    print(f"     Row {i}: {dict(row)}")
                    
        except Exception as e:
            print(f"   Error reading CSV: {e}")
    
    # Check for image_conditional_dataset.py
    dataset_py = data_path / "image_conditional_dataset.py"
    if dataset_py.exists():
        print(f"\n📄 image_conditional_dataset.py found!")
        print(f"   Size: {dataset_py.stat().st_size} bytes")

def analyze_png_files(data_dir):
    """Analyze PNG files to understand radio map format."""
    print("\n" + "="*60)
    print("ANALYZING PNG FILES (RADIO MAPS)")
    print("="*60)
    
    png_dir = Path(data_dir) / "png"
    if not png_dir.exists():
        print("❌ No png/ directory found")
        return None
    
    png_files = list(png_dir.glob("*.png"))
    print(f"Found {len(png_files)} PNG files")
    
    if len(png_files) == 0:
        print("❌ No PNG files found")
        return None
    
    # Analyze a few sample PNG files
    sample_files = png_files[:3]
    radio_data = []
    
    for i, png_file in enumerate(sample_files):
        print(f"\nSample {i+1}: {png_file.name}")
        try:
            img = Image.open(png_file)
            print(f"   Size: {img.size}")
            print(f"   Mode: {img.mode}")
            print(f"   Format: {img.format}")
            
            # Convert to array to analyze values
            img_array = np.array(img)
            print(f"   Array shape: {img_array.shape}")
            print(f"   Data type: {img_array.dtype}")
            print(f"   Value range: {img_array.min()} to {img_array.max()}")
            print(f"   Mean: {img_array.mean():.4f}")
            
            # Convert to tensor
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # RGB image - convert to grayscale for radio map
                img_gray = Image.open(png_file).convert('L')
                img_array = np.array(img_gray)
                print(f"   Converted to grayscale: {img_array.shape}")
            
            # Normalize to [0, 1] range
            if img_array.max() > 1:
                img_normalized = img_array.astype(np.float32) / 255.0
            else:
                img_normalized = img_array.astype(np.float32)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_normalized).unsqueeze(0)  # Add channel dimension
            radio_data.append(img_tensor)
            
            print(f"   Tensor shape: {img_tensor.shape}")
            print(f"   Tensor range: {img_tensor.min():.4f} to {img_tensor.max():.4f}")
            
        except Exception as e:
            print(f"   ❌ Error loading {png_file.name}: {e}")
    
    if radio_data:
        # Stack into batch
        batch_radio = torch.stack(radio_data, dim=0)
        print(f"\n✅ Created radio data batch: {batch_radio.shape}")
        return batch_radio
    
    return None

def analyze_conditioning_data(data_dir):
    """Try to understand conditioning data format (buildings, transmitters, cars)."""
    print("\n" + "="*60)
    print("ANALYZING CONDITIONING DATA")
    print("="*60)
    
    data_path = Path(data_dir)
    
    # Check polygon directory (might contain building data)
    polygon_dir = data_path / "polygon"
    if polygon_dir.exists():
        polygon_files = list(polygon_dir.glob("*"))
        print(f"📁 polygon/ directory: {len(polygon_files)} files")
        
        for file in polygon_files[:3]:
            print(f"   📄 {file.name}")
            if file.suffix == '.json':
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    print(f"      JSON keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                except Exception as e:
                    print(f"      Error reading JSON: {e}")
    
    # Check antenna directory (might contain transmitter data)
    antenna_dir = data_path / "antenna"
    if antenna_dir.exists():
        antenna_files = list(antenna_dir.glob("*"))
        print(f"\n📁 antenna/ directory: {len(antenna_files)} files")
        
        for file in antenna_files[:3]:
            print(f"   📄 {file.name}")
            try:
                if file.suffix == '.csv':
                    df = pd.read_csv(file)
                    print(f"      CSV: {len(df)} rows, columns: {list(df.columns)}")
                elif file.suffix == '.json':
                    with open(file, 'r') as f:
                        data = json.load(f)
                    print(f"      JSON keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            except Exception as e:
                print(f"      Error reading: {e}")
    
    # For now, create dummy conditioning data
    print(f"\n⚠️  Creating dummy conditioning data for testing")
    print(f"   You'll need to adapt this to your actual conditioning format")
    
    # Create 3-channel conditioning (buildings, transmitters, cars)
    dummy_conditioning = torch.rand(3, 3, 256, 256)  # 3 samples, 3 channels, 256x256
    print(f"   Dummy conditioning shape: {dummy_conditioning.shape}")
    
    return dummy_conditioning

def test_custom_dataset_loader(data_dir):
    """Try to use your custom image_conditional_dataset.py"""
    print("\n" + "="*60)
    print("TESTING CUSTOM DATASET LOADER")
    print("="*60)
    
    try:
        # Import your custom dataset
        sys.path.insert(0, data_dir)
        import image_conditional_dataset
        print("✅ Successfully imported image_conditional_dataset.py")
        
        # Inspect the module
        print(f"Module attributes: {[attr for attr in dir(image_conditional_dataset) if not attr.startswith('_')]}")
        
        # Try to find dataset classes
        for attr_name in dir(image_conditional_dataset):
            attr = getattr(image_conditional_dataset, attr_name)
            if isinstance(attr, type) and 'dataset' in attr_name.lower():
                print(f"Found dataset class: {attr_name}")
                
                try:
                    # Try to instantiate
                    dataset = attr(data_dir)
                    print(f"✅ Created dataset: {len(dataset)} samples")
                    
                    # Try to get a sample
                    if len(dataset) > 0:
                        sample = dataset[0]
                        print(f"Sample type: {type(sample)}")
                        if isinstance(sample, (list, tuple)):
                            print(f"Sample structure: {len(sample)} elements")
                            for i, elem in enumerate(sample):
                                if isinstance(elem, torch.Tensor):
                                    print(f"  Element {i}: tensor {elem.shape}")
                                elif isinstance(elem, dict):
                                    print(f"  Element {i}: dict with keys {list(elem.keys())}")
                                else:
                                    print(f"  Element {i}: {type(elem)}")
                    
                    return dataset
                    
                except Exception as e:
                    print(f"Failed to create {attr_name}: {e}")
        
        return None
        
    except Exception as e:
        print(f"❌ Failed to import custom dataset: {e}")
        return None

def main():
    """Main exploration function."""
    print("RADIO DATA EXPLORATION FOR CONDITIONED CONSISTENCY U-NET")
    print("Understanding your exact data format...")
    
    # Setup paths
    consistency_path, data_path = setup_paths()
    
    # Explore data structure
    explore_data_structure(data_path)
    
    # Analyze PNG files (radio maps)
    radio_data = analyze_png_files(data_path)
    
    # Analyze conditioning data
    conditioning_data = analyze_conditioning_data(data_path)
    
    # Test custom dataset loader
    custom_dataset = test_custom_dataset_loader(data_path)
    
    # Summary
    print("\n" + "="*60)
    print("EXPLORATION SUMMARY")
    print("="*60)
    
    if radio_data is not None:
        print(f"✅ Radio maps: Found PNG files, created batch {radio_data.shape}")
    else:
        print(f"❌ Radio maps: Could not load PNG files")
    
    if conditioning_data is not None:
        print(f"⚠️  Conditioning: Created dummy data {conditioning_data.shape}")
        print(f"   (You need to adapt this to your actual conditioning format)")
    else:
        print(f"❌ Conditioning: Could not understand conditioning format")
    
    if custom_dataset is not None:
        print(f"✅ Custom dataset: Successfully loaded")
    else:
        print(f"⚠️  Custom dataset: Could not load, may need manual integration")
    
    print(f"\nNext steps:")
    print(f"1. Review the exploration output above")
    print(f"2. Understand your exact conditioning format")
    print(f"3. Adapt the test to use your real conditioning data")
    print(f"4. Test the conditioned U-Net with your data")

if __name__ == "__main__":
    main()
