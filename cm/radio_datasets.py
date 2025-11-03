from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
warnings.filterwarnings("ignore")

# Additional imports for consistency models compatibility
import math
import random
from PIL import Image
import blobfile as bf
from mpi4py import MPI


class RadioUNet_c(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/Users/paria/RadioDiff/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            self.ind1=600
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        
        #normalize and stack input images as 3-channel RGB
        image_buildings=image_buildings/256
        image_Tx=image_Tx/256        
        #inputs
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, np.zeros((256,256))], axis=2)        
        else: #cars
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!
        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
        else:
            inputs=torch.from_numpy(inputs.transpose((2, 0, 1))).float()
            image_gain=torch.from_numpy(image_gain.transpose((2, 0, 1))).float()
            
        inputs=self.transform_GY(inputs)
        #print("inputs_normalized:",inputs.max(),inputs.min(),inputs.shape)
        #image_gain.shape: [1,256,256]
        if image_gain.ndim == 3:
            if image_gain.shape[0] == 1:
                image_gain_normalized = (image_gain * 2.0 - 1.0).squeeze(0)  # Remove the first dimension
        
        return [inputs, image_gain_normalized]


class RadioUNet_s(Dataset):
    """RadioMapSeer Loader for sparse sample input and accurate buildings (RadioUNet_s)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/Users/paria/RadioDiff/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 data_samples=300,
                 fix_samples= 50,
                 num_samples_low=10, 
                 num_samples_high=100,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "IRT4", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            data_samples: Number of IRT4 samples.
            fix_samples: if 0, each sample input has a random number of measurement samples. Othnerwise fix_samples measurements per sample input.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10.
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
            
        Output:
            
        """
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="IRT4":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT4/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT4/"
        
        elif simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"  
        
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
        self.data_samples=data_samples
        self.fix_samples= fix_samples
        self.num_samples_low= num_samples_low 
        self.num_samples_high= num_samples_high
        
        self.transform= transform
        
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
        
        
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))  #Will be normalized later, after random seed is computed from it
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))/256 
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
        
        image_gain=image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
                                  # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
                                  # Important: when evaluating the accuracy, remember to devide the errors by 256!
                    
        #Saprse IRT4 samples, determenistic and fixed samples per map
        sparse_samples = np.zeros((self.width,self.height))
        seed_map=np.sum(image_buildings) # Each map has its fixed samples, independent of the transmitter location.
        np.random.seed(seed_map)       
        x_samples=np.random.randint(0, 255, size=self.data_samples)
        y_samples=np.random.randint(0, 255, size=self.data_samples)
        sparse_samples[x_samples,y_samples]= 1
        
        #input samples from the sparse gain samples
        input_samples = np.zeros((256,256))
        if self.fix_samples==0:
            num_in_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_in_samples=np.floor(self.fix_samples).astype(int)
            
        data_inds=range(self.data_samples)
        input_inds=np.random.permutation(data_inds)[0:num_in_samples[0]]      
        x_samples_in=x_samples[input_inds]
        y_samples_in=y_samples[input_inds]
        input_samples[x_samples_in,y_samples_in]= image_gain[x_samples_in,y_samples_in,0]
        
        #normalize image_buildings, after random seed computed from it as an int
        image_buildings=image_buildings/256
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, input_samples], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, input_samples, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!
        

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            sparse_samples = self.transform(sparse_samples).type(torch.float32)
            

        return [inputs, image_gain, sparse_samples]


# ================================================================================
# CONSISTENCY MODELS COMPATIBILITY LAYER
# ================================================================================

class RadioImageDataset(Dataset):
    """
    Wrapper around RadioUNet datasets for consistency models compatibility.
    This converts the radio dataset output to the format expected by consistency models,
    using the input conditioning (buildings, transmitters, etc.) as image conditioning.
    """
    
    def __init__(self,
                 radio_dataset,
                 resolution,
                 use_image_conditioning=True,
                 classes=None,
                 shard=0,
                 num_shards=1,
                 random_crop=False,
                 random_flip=False):
        super().__init__()
        self.radio_dataset = radio_dataset
        self.resolution = resolution
        self.use_image_conditioning = use_image_conditioning
        self.local_classes = classes
        self.random_crop = random_crop
        self.random_flip = random_flip
        
        # Handle distributed training sharding (like original ImageDataset)
        total_length = len(radio_dataset)
        indices = list(range(total_length))
        self.local_indices = indices[shard::num_shards]
        
    def __len__(self):
        return len(self.local_indices)

    def __getitem__(self, idx):
        # Get actual index in the radio dataset
        real_idx = self.local_indices[idx]
        
        # Get data from your radio dataset
        data = self.radio_dataset[real_idx]
        
        # Extract both the target (image_gain) and conditioning (inputs)
        if len(data) == 3:
            inputs, image_gain, sparse_samples = data
        elif len(data) == 2:
            inputs, image_gain = data
        else:
            # Fallback - no conditioning available
            image_gain = data
            inputs = None
        
        # Process the target image (radio gain map)
        target_arr = self._process_image(image_gain, is_target=True)
        
        # Process the conditioning inputs if available and requested
        conditioning = {}
        if self.use_image_conditioning and inputs is not None:
            # Convert inputs to numpy if tensor
            if torch.is_tensor(inputs):
                inputs_arr = inputs.numpy()
            else:
                inputs_arr = np.array(inputs)
            
            # inputs should be in CHW format from your dataset
            # Resize conditioning to match target resolution
            conditioning_processed = self._process_conditioning(inputs_arr)
            conditioning["conditioning"] = conditioning_processed
        
        # Add class labels if provided (can be used alongside image conditioning)
        if self.local_classes is not None:
            conditioning["y"] = np.array(self.local_classes[real_idx], dtype=np.int64)
        
        return target_arr, conditioning
    
    def _process_image(self, image, is_target=True):
        """Process radio gain map or other single-channel images"""
        # Convert to numpy array
        if torch.is_tensor(image):
            arr = image.numpy()
        else:
            arr = np.array(image)
        
        # Handle different shapes - get single channel image
        if arr.ndim == 3:
            if arr.shape[0] == 1:  # CHW format with 1 channel
                arr = arr[0]  # Remove channel dimension -> HW
            elif arr.shape[2] == 1:  # HWC format with 1 channel  
                arr = arr[:, :, 0]  # Remove channel dimension -> HW
        elif arr.ndim == 2:
            # Already 2D, good
            pass
        else:
            raise ValueError(f"Unexpected image shape: {arr.shape}")
        
        # Ensure 2D
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {arr.shape}")
        
        # Apply transformations similar to original ImageDataset
        # Clamp values to [0, 1] range first
        arr = np.clip(arr, 0, 1)
        pil_image = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
        
        # Resize/crop like original consistency models
        if self.random_crop and is_target:  # Only apply random crop to target
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        # Random flip (apply same transformation to both target and conditioning)
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # Normalize to [-1, 1] range (exactly like original ImageDataset)
        arr = arr.astype(np.float32) / 127.5 - 1

        # Convert to CHW format (add channel dimension)
        arr = np.expand_dims(arr, axis=0)  # Shape: (1, H, W)
        
        return arr
    
    def _process_conditioning(self, inputs_arr):
        """Process the multi-channel conditioning inputs (buildings, transmitters, etc.)"""
        # inputs_arr should be in CHW format: (3 or 4, H, W)
        # Channel 0: Buildings
        # Channel 1: Transmitters  
        # Channel 2: Cars or zeros
        # Channel 3: Sparse samples (for RadioUNet_s) - optional
        
        if inputs_arr.ndim != 3:
            raise ValueError(f"Expected 3D conditioning input (CHW), got shape {inputs_arr.shape}")
        
        # Process each channel separately and resize
        processed_channels = []
        
        for i in range(inputs_arr.shape[0]):  # For each channel
            channel = inputs_arr[i]  # Shape: (H, W)
            
            # Convert to PIL for resizing
            # Clamp values to [0, 1] range
            channel_clamped = np.clip(channel, 0, 1)
            pil_channel = Image.fromarray((channel_clamped * 255).astype(np.uint8), mode='L')
            
            # Resize to target resolution
            channel_resized = center_crop_arr(pil_channel, self.resolution)
            
            # Apply same random flip as target if enabled
            if self.random_flip and hasattr(self, '_flip_applied') and self._flip_applied:
                channel_resized = channel_resized[:, ::-1]
            
            # Normalize to [-1, 1] range
            channel_normalized = channel_resized.astype(np.float32) / 127.5 - 1
            
            processed_channels.append(channel_normalized)
        
        # Stack channels back together: (C, H, W)
        conditioning_tensor = np.stack(processed_channels, axis=0)
        
        return conditioning_tensor


def center_crop_arr(pil_image, image_size):
    """Center crop function from original consistency models"""
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """Random crop function from original consistency models"""
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
    # Radio-specific parameters
    dataset_type="RadioUNet_c",
    phase="train",
    simulation="DPM",
    carsSimul="no",
    carsInput="no",
    cityMap="complete",
    missing=1,
    numTx=80,
    thresh=0.05,
    IRT2maxW=1,
    # For RadioUNet_s
    data_samples=300,
    fix_samples=50,
    num_samples_low=10,
    num_samples_high=100,
    # Conditioning options
    use_image_conditioning=True,  # Use rich input conditioning instead of just class labels
):
    """
    Create a generator over (images, kwargs) pairs for radio data.
    
    This is a drop-in replacement for the consistency models load_data function.
    
    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    
    For radio data, kwargs can contain:
    - "conditioning": Multi-channel conditioning images (buildings, transmitters, etc.)
    - "y": Class labels (optional, for simulation type)

    :param data_dir: path to the RadioMapSeer dataset directory
    :param batch_size: the batch size of each returned pair
    :param image_size: the size to which images are resized
    :param class_cond: if True, include a "y" key in returned dicts for class label
    :param deterministic: if True, yield results in a deterministic order
    :param random_crop: if True, randomly crop the images for augmentation
    :param random_flip: if True, randomly flip the images for augmentation
    :param dataset_type: "RadioUNet_c" or "RadioUNet_s"
    :param phase: "train", "val", "test"
    :param simulation: "DPM", "IRT2", "IRT4", "rand"
    :param carsSimul: "yes", "no" 
    :param carsInput: "yes", "no"
    :param cityMap: "complete", "missing", "rand"
    :param missing: 1-4 for missing buildings
    :param numTx: number of transmitters
    :param thresh: threshold parameter
    :param IRT2maxW: max weight for random simulation
    :param use_image_conditioning: if True, include conditioning images in kwargs
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    # Create the radio dataset
    radio_kwargs = {
        'phase': phase,
        'dir_dataset': data_dir,
        'numTx': numTx,
        'thresh': thresh,
        'simulation': simulation,
        'carsSimul': carsSimul,
        'carsInput': carsInput,
        'cityMap': cityMap,
        'missing': missing,
        'IRT2maxW': IRT2maxW,
        'transform': transforms.ToTensor()
    }
    
    # Add RadioUNet_s specific parameters
    if dataset_type == "RadioUNet_s":
        radio_kwargs.update({
            'data_samples': data_samples,
            'fix_samples': fix_samples,
            'num_samples_low': num_samples_low,
            'num_samples_high': num_samples_high,
        })
    
    # Create the appropriate radio dataset
    if dataset_type == "RadioUNet_c":
        radio_dataset = RadioUNet_c(**radio_kwargs)
    elif dataset_type == "RadioUNet_s":
        radio_dataset = RadioUNet_s(**radio_kwargs)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Create class labels if needed (can be used alongside image conditioning)
    classes = None
    if class_cond:
        # Simple class labeling based on simulation type
        class_map = {"DPM": 0, "IRT2": 1, "IRT4": 2, "rand": 3}
        class_label = class_map.get(simulation, 0)
        # Create class labels for all samples
        classes = [class_label] * len(radio_dataset)
    
    # Create the dataset wrapper with image conditioning support
    dataset = RadioImageDataset(
        radio_dataset=radio_dataset,
        resolution=image_size,
        use_image_conditioning=use_image_conditioning,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    
    # Create DataLoader (exactly like original load_data)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    
    # Return generator (exactly like original load_data)
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """Helper function from original consistency models (not used but kept for compatibility)"""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results
