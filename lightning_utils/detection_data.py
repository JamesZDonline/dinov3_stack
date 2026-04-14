#Base Packages
import os
import pathlib
from tqdm import tqdm
import matplotlib.colors as mcolors
import random



from typing import Iterable, Sequence

#Data Packages
import geopandas as gpd

#Rastervision Packages
# from rastervision.pytorch_learner.object_detection_utils import collate_fn as od_collate
from rastervision.core.data import ClassConfig, RasterioSource
from rastervision.pytorch_learner import ObjectDetectionRandomWindowGeoDataset,ObjectDetectionSlidingWindowGeoDataset, BoxList


#Deep Learning Packages
import torch
from torch.utils.data import ConcatDataset, DataLoader

import lightning as L


#Custom Packages
from geopacha_utilities.utilities import find_pixel_size



def od_collate(data: Iterable[Sequence]) -> tuple[torch.Tensor, list[BoxList]] | None:
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
    def __init__(self, train_ds_list, val_ds_list, batch_size=8, num_workers=0):
        super().__init__()
        self.train_ds_list = train_ds_list
        self.val_ds_list = val_ds_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Combine the lists of datasets into single ConcatDatasets
        if stage == "fit" or stage is None:
            self.train_dataset = ConcatDataset(self.train_ds_list)
            self.val_dataset = ConcatDataset(self.val_ds_list)
        
        if stage == "test":
            self.val_dataset = ConcatDataset(self.val_ds_list)

    def train_dataloader(self):
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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=od_collate,
            pin_memory=True
        )


class DebugObjectDetectionRandomWindowGeoDataset(ObjectDetectionRandomWindowGeoDataset):
    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except Exception as e:
            import traceback
            print(f"!!! __getitem__ failed on {self.scene.id}, idx {idx}:\n{traceback.format_exc()}", flush=True)
            return None

class ObjectDetectionDataFactory():
    def __init__(self,image_dir,labels,patch_dim,upsample_factor,class_config,augmentation_transform):
        self.image_dir = image_dir
        self.labels =labels
        self.patch_dim = patch_dim
        self.upsample_factor = upsample_factor
        self.class_config = class_config
        self.augmentation_transform = augmentation_transform

    def get_dataset_list(self,aoi_dir, dtype='training',test_code=False):
        dataset_list = []
        file_list = os.listdir(aoi_dir)
        if test_code: file_list=file_list[0:3]
        for aoi_file in tqdm(file_list, desc=f"making {dtype} dataset"):
            full_aoi_path = os.path.join(aoi_dir,aoi_file)

            if dtype == 'training':
                ds = self.get_random_window_geodataset(full_aoi_path)
            else:
                ds = self.get_sliding_window_geodataset(full_aoi_path)
            if ds is not None:          
                dataset_list.append(ds)
        return dataset_list

        
    def get_aoi_properties(self, full_aoi_path):
        
        try:
            aoi = gpd.read_file(full_aoi_path)
        except Exception as e:
            raise ValueError(f"Failed to read AOI file {full_aoi_path}: {e}") from e


        image_id = aoi['imageid'][0]
        image_path  = pathlib.PureWindowsPath(aoi['filepath'][0]).as_posix()
        full_image_path = os.path.join(self.image_dir,image_path)

        #allow user to pass labels in a single file, or a directory of geojsons
        if os.path.isfile(self.labels):
            full_label_path=self.labels
        else:   full_label_path = os.path.join(self.labels,f"labels_{image_id}.geojson")
        
        rasterSource = RasterioSource(
            full_image_path,
            allow_streaming = True,
        )

        pixel_size = find_pixel_size(rasterSource.imagery_path)
        area_of_chip = (self.patch_dim*(.5/self.upsample_factor))**2
        size = round(self.patch_dim*(.5/self.upsample_factor)/pixel_size)
        aoi_projected = aoi.to_crs('ESRI:102033')['geometry']
        num_chips = int(aoi_projected.area/area_of_chip)
        max_windows= min(max([aoi['label_count'][0],10]),num_chips)

        bounds = aoi.to_crs('EPSG:3857').geometry.bounds
        aoi_width = bounds['maxx'].values[0] - bounds['minx'].values[0]
        aoi_height = bounds['maxy'].values[0] - bounds['miny'].values[0]
        size_in_meters = size*pixel_size

        # print(f"Dimensions = ({aoi_width:.0f}m x {aoi_height:.0f}m, chip={size_in_meters:.0f}m)")

        if aoi_width < size_in_meters or aoi_height < size_in_meters:
            raise ValueError(
                f"{image_id} AOI too narrow to fit a chip "
                f"({aoi_width:.0f}m x {aoi_height:.0f}m, chip={size_in_meters:.0f}m)"
            )



        
        # print(f"{image_id} will have ~{num_chips} chips")

        aoi_has_labels=aoi['label_count'][0]!=0
        return full_image_path, full_aoi_path, full_label_path,image_id,size,aoi_has_labels,max_windows



    def get_random_window_geodataset(self,full_aoi_path):
        
        try:
            full_image_path, full_aoi_path, full_label_path,image_id,size,aoi_has_labels,max_windows = self.get_aoi_properties(full_aoi_path=full_aoi_path)
        except Exception as e:
            print(f"Skipping {full_aoi_path} because of {e}")
            return None
        neg_ratio = 0.5
        ioa_thresh = 0.5
        within_aoi=True
        if not aoi_has_labels:
            neg_ratio=None
            within_aoi=False
            full_label_path = os.path.join(self.labels, os.listdir(self.labels)[0])
        print(image_id)
        try:
                ds = DebugObjectDetectionRandomWindowGeoDataset.from_uris(
                    image_uri=full_image_path,
                    aoi_uri=full_aoi_path,
                    label_vector_uri = full_label_path,
                    class_config=self.class_config,
                    image_raster_source_kw=dict(allow_streaming=True,channel_order=[4,2,1]),
                    max_windows=max_windows,
                    size_lims = [size,size+1],
                    out_size=self.patch_dim,
                    within_aoi=within_aoi, 
                    ioa_thresh = ioa_thresh,
                    neg_ratio=neg_ratio,
                    transform = self.augmentation_transform
                )
                ds.scene.id=image_id
                return ds
                # rasterio._loading.get_pipeline()
        except Exception as e: 
            print(f"Couldn't create dataset because:\n{e}")       

    def get_sliding_window_geodataset(self,full_aoi_path): 
        try:
            full_image_path, full_aoi_path, full_label_path,image_id,size,aoi_has_labels,max_windows = self.get_aoi_properties(full_aoi_path=full_aoi_path)
        except Exception as e:
            print(f"Skipping {full_aoi_path} because of {e}")
            return None
        if not aoi_has_labels:full_label_path = os.path.join(self.labels, os.listdir(self.labels)[0])
        try:
              ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
                    image_uri = full_image_path,
                    aoi_uri = full_aoi_path,
                    label_vector_uri = full_label_path,
                    class_config = self.class_config,
                    size = size,
                    stride = int(size*0.5),
                    out_size = self.patch_dim,
                    within_aoi=True,
                    image_raster_source_kw=dict(allow_streaming=True,channel_order=[4,2,1]),return_window=False

              )
              ds.scene.id = image_id
              return ds
        except Exception as e:
              print(f"Problem with {image_id} sliding window: \n {e}")
              


# at the bottom of detection_data.py

if __name__ == '__main__':
    from rastervision.core.data import ClassConfig
    import pandas as pd

    TRAINING_AOI_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/FirstRun/AOI_Training'
    VALIDATION_AOI_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/FirstRun/AOI_Validation'
    LABEL_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/FirstRun/Labels'
    IMAGERY_BASE_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images'

    

    print("Read labels and create class config")
    label_dataframes=[]
    for label_file in tqdm(os.listdir(LABEL_DIRECTORY)):
        label_dataframes.append(gpd.read_file(os.path.join(LABEL_DIRECTORY,label_file)))
    labels = pd.concat(label_dataframes)

    # Get unique pairs and sort by class_id to ensure order
    mapping_df = labels[['class_id', 'class_name']].drop_duplicates().sort_values('class_id')

    names_list = mapping_df['class_name'].tolist()
    names_list.append("background")
    ids_list = mapping_df['class_id'].tolist()



    all_color_names = [c for c in mcolors.CSS4_COLORS.keys() if 'white' not in c and 'snow' not in c]
    colors_list = random.sample(all_color_names, len(names_list))


    class_config = ClassConfig(
        names=names_list,
        colors=colors_list,
        null_class='background')

    
    factory = ObjectDetectionDataFactory(
        image_dir=IMAGERY_BASE_DIRECTORY,
        labels=LABEL_DIRECTORY,
        patch_dim=1024,
        class_config=class_config,
        augmentation_transform=None
    )
    
    # Test just training data
    train_datasets = factory.get_dataset_list(
        aoi_dir=TRAINING_AOI_DIRECTORY,
        dtype='training'
    )
    
    print(f"\nGot {len(train_datasets)} valid training datasets")
    
    # Try actually pulling a chip from each dataset
    for ds in train_datasets:
        try:
            sample = ds[0]
            img, labels = sample
            print(f"{ds.scene.id}: chip shape {img.shape}, {len(labels.boxes)} boxes")
        except Exception as e:
            print(f"{ds.scene.id}: FAILED at __getitem__ — {e}")