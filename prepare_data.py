"""
Prepare RadioMapSeer dataset for consistency model training.
Extracts gain images from the dataset structure and organizes them 
for unconditional generation training.
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import io
import argparse

def prepare_radiomapseer_data(
    dataset_dir,
    output_dir,
    simulation="DPM",
    cars_simul="no",
    num_tx=80,
    maps_range=(0, 699),
    image_size=256
):
    """
    Extract and organize RadioMapSeer gain images for consistency model training.
    
    Args:
        dataset_dir: Path to RadioMapSeer dataset directory
        output_dir: Output directory for organized images
        simulation: "DPM", "IRT2", or "IRT4"
        cars_simul: "no" or "yes" - whether to use car simulation data
        num_tx: Number of transmitters per map (default 80)
        maps_range: Tuple of (start_map, end_map) indices
        image_size: Target image size (default 256)
    """
    
    # Set up gain directory path based on simulation type
    if simulation == "DPM":
        gain_dir = "gain/carsDPM/" if cars_simul == "yes" else "gain/DPM/"
    elif simulation == "IRT2":
        gain_dir = "gain/carsIRT2/" if cars_simul == "yes" else "gain/IRT2/"
    elif simulation == "IRT4":
        gain_dir = "gain/carsIRT4/" if cars_simul == "yes" else "gain/IRT4/"
    else:
        raise ValueError(f"Unknown simulation type: {simulation}")
    
    gain_path = os.path.join(dataset_dir, gain_dir)
    
    # Create output directory structure
    # For unconditional training, we put all images in a single class folder
    output_path = Path(output_dir) / "unconditional"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle maps deterministically (same as in your loader)
    maps_inds = np.arange(maps_range[0], maps_range[1] + 1, dtype=np.int16)
    np.random.seed(42)
    np.random.shuffle(maps_inds)
    
    image_count = 0
    
    print(f"Processing {len(maps_inds)} maps with {num_tx} transmitters each...")
    
    for map_idx in maps_inds:
        dataset_map_ind = map_idx + 1
        
        for tx_idx in range(num_tx):
            # Construct filename following your loader's naming convention
            filename = f"{dataset_map_ind}_{tx_idx}.png"
            gain_file_path = os.path.join(gain_path, filename)
            
            if not os.path.exists(gain_file_path):
                print(f"Warning: File not found: {gain_file_path}")
                continue
            
            try:
                # Load and process the gain image
                image_gain = io.imread(gain_file_path)
                
                # Convert to PIL Image for consistency with other datasets
                if len(image_gain.shape) == 2:
                    # Convert grayscale to RGB for consistency
                    image_gain_rgb = np.stack([image_gain] * 3, axis=-1)
                else:
                    image_gain_rgb = image_gain
                
                # Ensure correct size
                pil_image = Image.fromarray(image_gain_rgb.astype(np.uint8))
                if pil_image.size != (image_size, image_size):
                    pil_image = pil_image.resize((image_size, image_size), Image.LANCZOS)
                
                # Save with a unique filename
                output_filename = f"gain_{dataset_map_ind:04d}_{tx_idx:02d}.png"
                output_file_path = output_path / output_filename
                pil_image.save(output_file_path)
                
                image_count += 1
                
                if image_count % 1000 == 0:
                    print(f"Processed {image_count} images...")
                    
            except Exception as e:
                print(f"Error processing {gain_file_path}: {e}")
                continue
    
    print(f"Successfully processed {image_count} images")
    print(f"Output directory: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Prepare RadioMapSeer data for consistency models")
    parser.add_argument("dataset_dir", help="Path to RadioMapSeer dataset directory")
    parser.add_argument("output_dir", help="Output directory for processed images")
    parser.add_argument("--simulation", choices=["DPM", "IRT2", "IRT4"], default="DPM",
                        help="Simulation type to use")
    parser.add_argument("--cars-simul", choices=["no", "yes"], default="no",
                        help="Whether to use car simulation data")
    parser.add_argument("--num-tx", type=int, default=80,
                        help="Number of transmitters per map")
    parser.add_argument("--start-map", type=int, default=0,
                        help="Starting map index")
    parser.add_argument("--end-map", type=int, default=699,
                        help="Ending map index")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Target image size")
    
    args = parser.parse_args()
    
    prepare_radiomapseer_data(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        simulation=args.simulation,
        cars_simul=args.cars_simul,
        num_tx=args.num_tx,
        maps_range=(args.start_map, args.end_map),
        image_size=args.image_size
    )

if __name__ == "__main__":
    main()
