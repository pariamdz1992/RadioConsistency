from __future__ import print_function, division
import math
import random
import os

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from skimage import io, transform
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_radio_data(
    *,
    data_dir,
    batch_size, 
    image_size=256,
    dataset_variant="radio",
    phase="train",
    deterministic=False,
):
    """
    Load radio map data with simple spatial conditioning for consistency models.
    
    :param data_dir: path to RadioMapSeer dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size of radio maps (should be 256).
    :param dataset_variant: which radio map variant ("radio", "radio_irt4").
    :param phase: "train", "val", or "test".
    :param deterministic: if True, yield results in deterministic order.
    """
    dataset = RadioDataset(
        data_dir=data_dir,
        image_size=image_size,
        dataset_variant=dataset_variant,
        phase=phase,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    
    while True:
        yield from loader


class RadioDataset(Dataset):
    """RadioMap dataset adapted for consistency models with simple spatial conditioning"""
    def __init__(
        self,
        data_dir,
        image_size=256,
        dataset_variant="radio", 
        phase="train",
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.dataset_variant = dataset_variant
        
        # Create the radio map dataset based on variant
        if dataset_variant == "radio":
            self.radio_dataset = RadioUNet_c(
                phase=phase,
                dir_dataset=data_dir,
                simulation="DPM",
                carsSimul="no", 
                carsInput="no",
                transform=transforms.ToTensor()
            )
        elif dataset_variant == "radio_irt4":
            self.radio_dataset = RadioUNet_c_sprseIRT4(
                phase=phase,
                dir_dataset=data_dir,
                simulation="IRT4",
                carsSimul="no",
                carsInput="no",
                numTx=2,
                transform=transforms.ToTensor()
            )
        else:
            raise ValueError(f"Unknown dataset_variant: {dataset_variant}")
        
        # Handle distributed training sharding
        total_length = len(self.radio_dataset)
        items_per_shard = total_length // num_shards
        start_idx = shard * items_per_shard
        if shard == num_shards - 1:  # Last shard gets remainder
            end_idx = total_length
        else:
            end_idx = start_idx + items_per_shard
        
        self.indices = list(range(start_idx, end_idx))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        if self.dataset_variant == "radio":
            inputs, image_gain = self.radio_dataset[actual_idx]
        elif self.dataset_variant == "radio_irt4":
            inputs, image_gain, image_samples = self.radio_dataset[actual_idx]
        
        # inputs: [C, H, W] spatial conditions (buildings, transmitters)
        # image_gain: [1, H, W] target radio map
        
        # Ensure image_gain is in [-1, 1] range (consistency model format)
        if image_gain.max() <= 1.0 and image_gain.min() >= 0.0:
            image_gain = image_gain * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        # Prepare output dictionary with simple spatial conditioning
        # inputs is already [C, H, W] with C=2 (buildings + transmitters)
        out_dict = {"cond": inputs}  # Simple spatial conditioning
        
        return image_gain, out_dict


class RadioUNet_c(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/data/",
                 numTx=80,                  
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform=transforms.ToTensor()):
        
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
            self.dir_buildings = self.dir_dataset+"png/buildings_missing"
            
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
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
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
                 
        # Simple inputs to radioUNet - normalize to [0,1] for better training
        if self.carsInput=="no":
            # Normalize buildings and transmitters to [0,1]
            buildings_norm = image_buildings / 255.0
            tx_norm = image_Tx / 255.0
            inputs = np.stack([buildings_norm, tx_norm], axis=2)        
        else: # cars
            # Normalize all inputs to [0,1]
            buildings_norm = image_buildings / 255.0
            tx_norm = image_Tx / 255.0
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars)) / 255.0
            inputs = np.stack([buildings_norm, tx_norm, image_cars], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)

        return [inputs, image_gain]


class RadioUNet_c_sprseIRT4(Dataset):
    """RadioMapSeer Loader for IRT4 sparse data"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/data/",
                 numTx=2,                  
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="no",
                 carsInput="no",
                 cityMap="complete",
                 missing=1,
                 num_samples=300,
                 transform=transforms.ToTensor()):
        
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
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
        else:
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
            self.dir_buildings = self.dir_dataset+"png/buildings_missing"
            
        self.transform= transform
        self.num_samples=num_samples
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
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
        name1 = str(dataset_map_ind) + ".png"
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
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
        
        #Sparse IRT4 samples
        image_samples = np.zeros((self.width,self.height))
        seed_map=np.sum(image_buildings)
        np.random.seed(seed_map)       
        x_samples=np.random.randint(0, 255, size=self.num_samples)
        y_samples=np.random.randint(0, 255, size=self.num_samples)
        image_samples[x_samples,y_samples]= 1
        
        # Simple inputs to radioUNet - normalize to [0,1] for better training
        if self.carsInput=="no":
            # Normalize buildings and transmitters to [0,1]
            buildings_norm = image_buildings / 255.0
            tx_norm = image_Tx / 255.0
            inputs = np.stack([buildings_norm, tx_norm], axis=2)        
        else:
            # Normalize all inputs to [0,1]
            buildings_norm = image_buildings / 255.0
            tx_norm = image_Tx / 255.0
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars)) / 255.0
            inputs = np.stack([buildings_norm, tx_norm, image_cars], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            image_samples = self.transform(image_samples).type(torch.float32)

        return [inputs, image_gain, image_samples]


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
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
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

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
