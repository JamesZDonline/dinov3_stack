from rastervision.pytorch_learner import DataConfig, GeoDataset
from tqdm import tqdm
import os
from rastervision.pytorch_learner import ObjectDetectionRandomWindowGeoDataset,ObjectDetectionSlidingWindowGeoDataset
from rastervision.pytorch_learner.object_detection_utils import collate_fn as od_collate
from rastervision.core.data import ObjectDetectionLabelSourceConfig
from rastervision.core.data import ObjectDetectionLabelSource
from torch.utils.data import ConcatDataset
from rastervision.core.data import GeoJSONVectorSource, RasterioCRSTransformer,ClassConfig
from rastervision.pytorch_learner import ObjectDetectionModelConfig
from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter

import os
import geopandas as gpd

import pathlib

from rastervision.core.data import GeoJSONVectorSource, RasterioCRSTransformer,ClassConfig
from rastervision.pytorch_learner import ClassificationSlidingWindowGeoDataset
from rastervision.core.data import RasterioSource
from geopacha_utilities.utilities import find_pixel_size
AOI_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3-james-experiments/OD_data/Buffered_Chacu_AOI/SecondPass'
LABEL_URI = '/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3-james-experiments/OD_data/chacu_labels_first_pass.geojson'
IMAGERY_BASE_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images'

import albumentations as A

data_augmentation_transform = A.Compose([
    A.D4(p=1.0),
    # A.OneOf([
    #     A.HueSaturationValue(hue_shift_limit=10),
    #     A.RGBShift(),
    #     A.ToGray(),
    #     A.ToSepia(),
    #     A.RandomBrightnessContrast(),
    #     A.RandomGamma(),
    # ]),
    # A.CoarseDropout(max_height=32, max_width=32, max_holes=5)
])

patch_dim = 1024

# Your existing model definition parameters
repo_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3"
weights_path = "/home/VANDERBILT/zimmejr1/Documents/DinoV3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
model_name = "dinov3_vitl16"



class DinoV3DetectionModelConfig(ObjectDetectionModelConfig):
    # This build method is what the Learner calls to create the model
    # inside each DDP process.
    def build(self, class_map=None, model_path=None,num_classes=None,in_channels=None,save_dir=None,hubconf_dir=None,img_sz=None,ddp_rank=None):
        from src.detection.model import dinov3_detection

        # 1. Instantiate the base DinoV3 model
        base_model = dinov3_detection(
            fine_tune=True,
            num_classes=num_classes, # Use the class_map to get num_classes dynamically
            weights=weights_path,
            model_name=model_name,
            repo_dir=repo_path,
            feature_extractor="multi",
            head="retinanet"
        )
        
        # 2. Wrap it with the TorchVisionODAdapter
        model = TorchVisionODAdapter(base_model)
        
        return model

class CustomObjectDetectionDataConfig(DataConfig):
    ### NOTE! THIS USES THE SAME DATA FOR TRAINING AND VALIDATION. THIS IS BAD, BUT I'M JUST TESTING THE CONCEPT. IT WILL NEED TO BE CHANGED FOR REAL USE!!!
    class_config: ClassConfig
    num_workers: int = 1

    aoi_directory:str = AOI_DIRECTORY
    imagery_base_directory: str = IMAGERY_BASE_DIRECTORY
    label_uri: str = LABEL_URI
    patch_dim: int = patch_dim

    def build(self, 
              tmp_dir = None, 
              **kwargs):
        
        # 1. Build the training dataset by calling your implemented method
        train_ds = self.build_dataset(split='train')
        
        # 2. Build the validation dataset
        valid_ds = self.build_dataset(split='valid')
        
        # 3. (Optional) Build the test dataset (assuming you don't use it now)
        test_ds = None 
        # If you were using test data, you'd add: test_ds = self.build_dataset(split='test')
        
        # Return the required tuple: (train_ds, valid_ds, test_ds)
        return train_ds, valid_ds, test_ds

    def build_dataset(self, split: str):
        is_train = split =='train'
        
        dataset_list= []

        for aoi_path in tqdm(os.listdir(AOI_DIRECTORY),desc="making datasets"):
            full_aoi_path = os.path.join(AOI_DIRECTORY,aoi_path)
            aoi = gpd.read_file(full_aoi_path)
            image_id = aoi['imageid'][0]
            # print(image_id)
            image_path  = pathlib.PureWindowsPath(aoi['filepath'][0]).as_posix()
            full_image_path = os.path.join(IMAGERY_BASE_DIRECTORY,image_path)

            #Adjust the patch size to account for different resolution
            rasterSource = RasterioSource(
                    full_image_path, #path to the image
                    allow_streaming=True, # allow_streaming so we don't have to load the whole image
                ) 
            pixel_size = find_pixel_size(rasterSource.imagery_path)
            size = round(patch_dim*.5/pixel_size)
            if is_train:
                try:
                    ds = ObjectDetectionRandomWindowGeoDataset.from_uris(
                        image_uri=full_image_path,
                        aoi_uri=full_aoi_path,
                        label_vector_uri = LABEL_URI,
                        class_config=self.class_config,
                        image_raster_source_kw=dict(allow_streaming=True,channel_order=[4,2,1]),
                        max_windows=4,
                        size_lims = [size,size+1],
                        # size=size,
                        # stride=size,
                        out_size=patch_dim,within_aoi=True,ioa_thresh = 0.75,neg_ratio=1,
                        transform = data_augmentation_transform
                    )
                    ds.scene.id=image_id
                    dataset_list.append(ds)
                except:
                    try:
                        # print("Extracting Negatives")
                        ds = ObjectDetectionRandomWindowGeoDataset.from_uris(
                            image_uri=full_image_path,
                            aoi_uri=full_aoi_path,
                            label_vector_uri = LABEL_URI,
                            class_config=self.class_config,
                            image_raster_source_kw=dict(allow_streaming=True,channel_order=[4,2,1]),
                            max_windows=2,
                            size_lims = [size,size+1],
                            # size=size,
                            # stride=size,
                            out_size=patch_dim,within_aoi=True,
                            transform = data_augmentation_transform
                        )
                        ds.scene.id=image_id
                        dataset_list.append(ds)
                    except Exception as e: 
                        print(f"Couldn't create dataset because:\n{e}")                   
                        continue
            elif split == 'valid':
                try:
                    ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
                    image_uri = full_image_path,
                    aoi_uri = full_aoi_path,
                    label_vector_uri = LABEL_URI,
                    class_config = self.class_config,
                    size = size,
                    stride = size,
                    out_size = patch_dim,within_aoi=True,
                    image_raster_source_kw=dict(allow_streaming=True,channel_order=[4,2,1]),return_window=False
                    )
                    ds.scene.id = image_id
                    dataset_list.append(ds)
                except Exception as e:
                    print(f"Problem with {image_id} sliding window: \n {e}")
                    continue
        return ConcatDataset(dataset_list)
            


