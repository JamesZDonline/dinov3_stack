# Base Packages
import os
import pathlib
from tqdm import tqdm
import matplotlib.colors as mcolors
import random

from typing import Iterable, Sequence

# Data Packages
import geopandas as gpd

# Rastervision Packages
from rastervision.core.data import ClassConfig, RasterioSource
from rastervision.pytorch_learner import (
    ObjectDetectionRandomWindowGeoDataset,
    ObjectDetectionSlidingWindowGeoDataset,
    BoxList
)

# Deep Learning Packages
import torch
from torch.utils.data import ConcatDataset, DataLoader

import lightning as L

# Custom Packages
from geopacha_utilities.utilities import find_pixel_size


def od_collate(data: Iterable[Sequence]) -> tuple[torch.Tensor, list[BoxList]] | None:
    """
    Collate function for object detection datasets.
    
    Filters out None samples from failed __getitem__ calls (occurs when feature is larger than the image window) and handles batch creation.
    
    Args:
        data: Iterable of dataset samples
        
    Returns:
        Tuple of (images tensor, list of BoxLists) or None if all samples failed
    """
    # Filter out any None samples that came from failed __getitem__ calls
    valid_data = [d for d in data if d is not None and d[0] is not None]
    
    if not valid_data:
        return None  # whole batch was bad
    
    if len(valid_data) < len(data):
        print(f"Warning: dropped {len(data) - len(valid_data)} bad samples from batch")
    
    imgs = [d[0] for d in valid_data]
    x = torch.stack(imgs)
    y: list[BoxList] = [d[1] for d in valid_data]
    return x, y


# Lightning Modules
class ArchaeologyDataModule(L.LightningDataModule):
    """
    Lightning DataModule for archaeology object detection.
    
    Handles training and validation data loading with proper dataset concatenation.
    """
    
    def __init__(self, train_ds_list, val_ds_list, batch_size=8, num_workers=0):
        super().__init__()
        self.train_ds_list = train_ds_list
        self.val_ds_list = val_ds_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """
        Setup datasets for different stages (fit, test).
        
        Args:
            stage: Stage name ('fit', 'test', or None)
        """
        # Combine the lists of datasets into single ConcatDatasets
        if stage == "fit" or stage is None:
            self.train_dataset = ConcatDataset(self.train_ds_list)
            self.val_dataset = ConcatDataset(self.val_ds_list)
        
        if stage == "test":
            self.val_dataset = ConcatDataset(self.val_ds_list)

    def train_dataloader(self):
        """
        Create training data loader.
        
        Returns:
            DataLoader for training data
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=od_collate,  # Crucial for Raster Vision OD datasets
            pin_memory=True,
            persistent_workers=True,  # keeps workers alive between epochs
        )

    def val_dataloader(self):
        """
        Create validation data loader.
        
        Returns:
            DataLoader for validation data
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=od_collate,
            pin_memory=True
        )


class DebugObjectDetectionRandomWindowGeoDataset(ObjectDetectionRandomWindowGeoDataset):
    """
    Debug wrapper for ObjectDetectionRandomWindowGeoDataset that catches and reports errors.
    
    This helps identify problematic data samples during training.
    """
    
    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except Exception as e:
            import traceback
            print(f"!!! __getitem__ failed on {self.scene.id}, idx {idx}:\n{traceback.format_exc()}", flush=True)
            return None


class ObjectDetectionDataFactory():
    """
    Factory class for creating object detection datasets.
    
    Creates training and validation datasets from AOI files and labels.

    """
    
    def __init__(self, image_dir, labels, patch_dim, upsample_factor, class_config, 
                 augmentation_transform, channel_order=[4, 2, 1]):
        """
        Initialize the factory with configuration parameters.
        
        Args:
            image_dir: Directory containing imagery files
            labels: Path to label files (single file or directory)
            patch_dim: Dimension of output patches
            upsample_factor: Upsample factor affects the size of chips extracted from imagery. 
                    For example upsample_factor=2 with a patch_dim of 512 means a 256x256 pixel 
                    area of the original image (128m x 128m at 0.5 m resolution) will be extracted 
                    and upsampled to 512x512 for training.
            class_config: RasterVision Class configuration for object detection
            augmentation_transform: Transformations to apply to images
            channel_order: Channel order for image loading (default: [4,2,1] corresponding to RGB for WV2/WV3)
        """
        self.image_dir = image_dir
        self.labels = labels
        self.patch_dim = patch_dim
        self.upsample_factor = upsample_factor
        self.class_config = class_config
        self.augmentation_transform = augmentation_transform
        self.channel_order = channel_order

    def get_dataset_list(self, aoi_dir, dtype='training', test_code=False):
        """
        Create list of pytorch datasets from AOI directory.
        - Training: Random window sampling from AOIs provides a diverse set of training samples.
        - Validation: Sliding window sampling from AOIs provides a comprehensive evaluation of the model across the entire AOI.
        - Other: Sliding window sampling can also be used for testing or inference on new AOIs.
        
        Args:
            aoi_dir: Directory containing AOI files
            dtype: Dataset type ('training', 'validation', or other)
            test_code: Flag to limit processing for testing
            
        Returns:
            List of created datasets
        """
        dataset_list = []
        file_list = os.listdir(aoi_dir)
        if test_code: 
            file_list = file_list[0:3]
            
        for aoi_file in tqdm(file_list, desc=f"making {dtype} dataset"):
            full_aoi_path = os.path.join(aoi_dir, aoi_file)

            if dtype == 'training':
                ds = self.get_random_window_geodataset(full_aoi_path)
            elif dtype == 'validation':
                ds = self.get_sliding_window_geodataset(full_aoi_path, stride_factor=1)
            else:
                ds = self.get_sliding_window_geodataset(full_aoi_path)
                
            if ds is not None:          
                dataset_list.append(ds)
        return dataset_list

        
    def get_aoi_properties(self, full_aoi_path):
        """
        Extract AOI properties for dataset creation.
        
        Args:
            full_aoi_path: Path to AOI file
            
        Returns:
            Tuple of (image_path, label_path, image_id, chip_size, has_labels, max_windows)
            
        Raises:
            ValueError: If AOI file cannot be read
        """
        try:
            aoi = gpd.read_file(full_aoi_path)
        except Exception as e:
            raise ValueError(f"Failed to read AOI file {full_aoi_path}: {e}") from e

        image_id = aoi['imageid'][0]
        image_path = pathlib.PureWindowsPath(aoi['filepath'][0]).as_posix()
        full_image_path = os.path.join(self.image_dir, image_path)

        # Allow user to pass labels in a single file, or a directory of geojsons
        if os.path.isfile(self.labels):
            full_label_path = self.labels
        else:   
            full_label_path = os.path.join(self.labels, f"labels_{image_id}.geojson")
        
        rasterSource = RasterioSource(
            full_image_path,
            allow_streaming=True,
        )

        pixel_size = find_pixel_size(rasterSource.imagery_path)
        area_of_chip = (self.patch_dim * (.5 / self.upsample_factor)) ** 2
        size = round(self.patch_dim * (.5 / self.upsample_factor) / pixel_size)
        
        aoi_projected = aoi.to_crs('ESRI:102033')['geometry']
        num_chips = int(aoi_projected.area / area_of_chip)
        max_windows = min(max([aoi['label_count'][0], 10]), num_chips)

        bounds = aoi.to_crs('EPSG:3857').geometry.bounds
        aoi_width = bounds['maxx'].values[0] - bounds['minx'].values[0]
        aoi_height = bounds['maxy'].values[0] - bounds['miny'].values[0]
        size_in_meters = size * pixel_size

        if aoi_width < size_in_meters or aoi_height < size_in_meters:
            raise ValueError(
                f"{image_id} AOI too narrow to fit a chip "
                f"({aoi_width:.0f}m x {aoi_height:.0f}m, chip={size_in_meters:.0f}m)"
            )

        aoi_has_labels = aoi['label_count'][0] != 0
        if not aoi_has_labels:
            max_windows = min(num_chips, 30)
            
        return full_image_path, full_aoi_path, full_label_path, image_id, size, aoi_has_labels, max_windows


    def get_random_window_geodataset(self, full_aoi_path):
        """
        Create random window dataset for training.
        
        Args:
            full_aoi_path: Path to AOI file
            
        Returns:
            Created dataset or None if creation failed
        """
        try:
            full_image_path, full_aoi_path, full_label_path, image_id, size, aoi_has_labels, max_windows = self.get_aoi_properties(full_aoi_path=full_aoi_path)
        except Exception as e:
            print(f"Skipping {full_aoi_path} because of {e}")
            return None
            
        neg_ratio = 0.5 # each epoch has a similar number of positive and negative samples, though the data overall is unbalanced.
        ioa_thresh = 0.5
        within_aoi = True
        
        if not aoi_has_labels:
            neg_ratio = None
            within_aoi = False
            full_label_path = os.path.join(self.labels, os.listdir(self.labels)[0])
            
        print(image_id)
        
        try:
            ds = DebugObjectDetectionRandomWindowGeoDataset.from_uris(
                image_uri=full_image_path,
                aoi_uri=full_aoi_path,
                label_vector_uri=full_label_path,
                class_config=self.class_config,
                image_raster_source_kw=dict(allow_streaming=True, channel_order=self.channel_order),
                max_windows=max_windows,
                size_lims=[size, size + 1],
                out_size=self.patch_dim,
                within_aoi=within_aoi, 
                ioa_thresh=ioa_thresh,
                neg_ratio=neg_ratio,
                transform=self.augmentation_transform
            )
            ds.scene.id = image_id
            return ds
            
        except Exception as e: 
            print(f"Couldn't create dataset because:\n{e}")       


    def get_sliding_window_geodataset(self, full_aoi_path, stride_factor=0.5): 
        """
        Create sliding window dataset for validation/testing.
        
        Args:
            full_aoi_path: Path to AOI file
            stride_factor: Factor determining how much overlap between windows (default: 0.5, reduce speed at potential cost to accuracy)
            
        Returns:
            Created dataset or None if creation failed
        """
        try:
            full_image_path, full_aoi_path, full_label_path, image_id, size, aoi_has_labels, max_windows = self.get_aoi_properties(full_aoi_path=full_aoi_path)
        except Exception as e:
            print(f"Skipping {full_aoi_path} because of {e}")
            return None
            
        if not aoi_has_labels:
            full_label_path = os.path.join(self.labels, os.listdir(self.labels)[0])
            
        try:
            ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
                image_uri=full_image_path,
                aoi_uri=full_aoi_path,
                label_vector_uri=full_label_path,
                class_config=self.class_config,
                size=size,
                stride=int(size * stride_factor),
                out_size=self.patch_dim,
                within_aoi=True,
                image_raster_source_kw=dict(allow_streaming=True, channel_order=self.channel_order),
                return_window=False
            )
            ds.scene.id = image_id
            return ds
            
        except Exception as e:
            print(f"Problem with {image_id} sliding window: \n {e}")
