from rastervision.core.data import ObjectDetectionLabels
import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import sigmoid_focal_loss
import os
import pathlib

from rastervision.core.data import GeoJSONVectorSource, RasterioCRSTransformer,ClassConfig
from rastervision.pytorch_learner import ClassificationSlidingWindowGeoDataset
from rastervision.core.data import RasterioSource
from geopacha_utilities.utilities import find_pixel_size
from src.detection.model import dinov3_detection
from tqdm import tqdm
from rastervision.pytorch_learner import ObjectDetectionRandomWindowGeoDataset,ObjectDetectionSlidingWindowGeoDataset
from rastervision.pytorch_learner.object_detection_utils import collate_fn as od_collate
from rastervision.core.data import ObjectDetectionLabelSourceConfig
from rastervision.core.data import ObjectDetectionLabelSource
#Base packages
import os
import pathlib
from tqdm import tqdm

#Data Packages
import pandas as pd
import geopandas as gpd

#Rastervision packages
from rastervision.core.data import GeoJSONVectorSource, RasterioCRSTransformer, ClassConfig, RasterioSource,ObjectDetectionLabelSourceConfig,ObjectDetectionLabelSource
from rastervision.pytorch_learner import ObjectDetectionRandomWindowGeoDataset,ObjectDetectionSlidingWindowGeoDataset
from rastervision.pytorch_learner.object_detection_utils import collate_fn as od_collate
from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter


#Deep Learning Packages
import torch
from torch.utils.data import DataLoader, ConcatDataset

from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.utils import draw_bounding_boxes

from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import Callback,TQDMProgressBar
from lightning.pytorch.tuner import Tuner


import albumentations as A


#Custom Packages
from geopacha_utilities.utilities import find_pixel_size
from geopacha_utilities.model_builder import get_calibrated_dinov3_model,ArchDetectionModule



os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

# TRAINING_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/training_aoi'
# VALIDATION_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/validation_aoi_sampled'
# LABEL_URI = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/chacu_labels.geojson'
IMAGERY_BASE_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images'
LABEL_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/FirstRun/Labels'

repo_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3"
weights_path = "/home/VANDERBILT/zimmejr1/Documents/DinoV3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
# weights_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3_stack/checkpoints/arch-dinov3-epoch=15-mAP50=0.192.ckpt"

model_name  = "dinov3_vitl16"



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

import matplotlib.colors as mcolors
import random

all_color_names = [c for c in mcolors.CSS4_COLORS.keys() if 'white' not in c and 'snow' not in c]
colors_list = random.sample(all_color_names, len(names_list))


class_config = ClassConfig(
    names=names_list,
    colors=colors_list,
    null_class='background')

# class_config = ClassConfig(
#     names=['chacu', 'background'],
#     colors=['darkred', 'gray'],
#     null_class='background')


#Constants

TRAINING_AOI_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/FirstRun/AOI_Training'
LABEL_URI = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/chacu_labels.geojson'
# IMAGERY_BASE_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images'

repo_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3"
# weights_path = "/home/VANDERBILT/zimmejr1/Documents/DinoV3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
model_name  = "dinov3_vitl16"

torch.set_float32_matmul_precision('high')

#Variables
patch_dim = 1024
batch_size = 16
num_workers = 0 #batch_size*2
learning_rate = 7.1e-5
# model_run_name = 'lr_7e-5_batch_16g2_longer_run'
smoothing=.01
gamma = 2
early_stop_patience = 20
upsample_factor = 2
# upsample_factor = 1




model = get_calibrated_dinov3_model(num_classes =len(class_config.names),
                                    weights_path=weights_path,
                                    model_name=model_name,
                                    repo_path=repo_path,
                                    smoothing=smoothing,
                                    gamma=gamma)

adapter_model = TorchVisionODAdapter(model)

model = ArchDetectionModule(
    model=adapter_model, 
    class_config=class_config, 
    lr=learning_rate
)
checkpoint=torch.load("checkpoints/luna_upsample_plateau2/arch-dinov3-epoch=32-mAP50=0.336.ckpt",map_location="cpu")

model.load_state_dict(checkpoint['state_dict'])

# # model = get_calibrated_dinov3_model(num_classes=2,smoothing=smoothing,gamma=gamma)
# from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter
# adapter_model = TorchVisionODAdapter(model)

# model = ArchDetectionModule(
#     model=adapter_model, 
#     class_config=class_config, 
#     lr=learning_rate
# )
# checkpoint=torch.load("checkpoints/chacu-dinov3-epoch=31-mAP50=0.539.ckpt",map_location="cpu")
# model.load_state_dict(checkpoint['state_dict'])


print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
from rastervision.pytorch_learner import ObjectDetectionLearner
from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import ConcatDataset, DataLoader
from rastervision.core.data import ObjectDetectionLabels
from rastervision.pytorch_learner.object_detection_utils import collate_fn as od_collate

def get_predictions(dataloader, model, device='cuda'):
    model.eval()
    model.to(device)
    for x, _ in tqdm(dataloader):
        with torch.inference_mode():
            # Move list of images to GPU
            x = [img.to(device) for img in x]

            ds = dataloader.dataset
            native_window_size = ds.size[0]
            model_input_size = ds.out_size[0]

            scale_factor = native_window_size/model_input_size
            
            # The Adapter returns a list of BoxList objects
            out_batch = model(x)
            
        # Yield each BoxList moved to CPU
        for out in out_batch:
                boxes = out.convert_boxes('yxyx').cpu().numpy()
                scaled_boxes = boxes*scale_factor
            # Convert the BoxList object to a CPU dictionary
            # Raster Vision's ObjectDetectionLabels expects:
            # {'boxes': np.array, 'class_ids': np.array, 'scores': np.array}
                yield {
                    'boxes': scaled_boxes,
                    'class_ids': out.get_field('class_ids').cpu().numpy(),
                    'scores': out.get_field('scores').cpu().numpy()
                }

# 1. Generate predictions
model.eval()

# AOI_DIRECTORY = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3-james-experiments/OD_data/Buffered_Chacu_AOI/ExpandedSecondPass"
# datasplit="Training"
# datasplit="Validation"
# AOI_DIRECTORY = f'/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/FirstRun/AOI_{datasplit}'
AOI_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3-james-experiments/OD_data/Chacha_AOI'
patch_dim=1024
for aoi_path in (os.listdir(AOI_DIRECTORY)):
      print("Making dataset")
      full_aoi_path = os.path.join(AOI_DIRECTORY,aoi_path)
      aoi = gpd.read_file(full_aoi_path)
      image_id_aoi = aoi['imageid'][0]
      image_id = image_id_aoi.split('_')[0]
    #   if image_id !='506412069030':continue
      output_path = os.path.join('/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3-james-experiments/OD_data/Chacha_output_full',f"{image_id}_chacu_labels2.geojson")
      print(image_id)

      if os.path.exists(output_path): continue
      image_path  = pathlib.PureWindowsPath(aoi['filepath'][0]).as_posix()
      full_image_path = os.path.join(IMAGERY_BASE_DIRECTORY,image_path)

      try:
        #Adjust the patch size to account for different resolution
        rasterSource = RasterioSource(
        full_image_path, #path to the image
        allow_streaming=True, # allow_streaming so we don't have to load the whole image
        ) 
        pixel_size = find_pixel_size(rasterSource.imagery_path)
        size = round(patch_dim*(.5/upsample_factor)/pixel_size)
        ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
              image_uri = full_image_path,
            #   aoi_uri = full_aoi_path,
              within_aoi=False,
              # label_vector_uri = LABEL_URI,
              class_config = class_config,
              size = size,
              stride = int(size/2),
              out_size = patch_dim,
              image_raster_source_kw=dict(allow_streaming=True,channel_order=[4,2,1]),return_window=False

              )
        ds.scene.id = image_id
        # predictions = learner.predict_dataset(
        #   ds,
        #   raw_out=True,
        #   numpy_out=True,
        #   progress_bar=True,
        #   dataloader_kw = {'num_workers':16,'batch_size':8})
        inference_dl = DataLoader(
            ds,
            batch_size=16, 
            shuffle=False,     # Never shuffle during inference
            num_workers=32,     # Parallel loading
            collate_fn=od_collate
        )
        predictions = get_predictions(inference_dl, model)
        

        pred_labels = ObjectDetectionLabels.from_predictions(
          ds.windows,
          predictions,
          )
        pred_labels_2 = pred_labels.prune_duplicates(pred_labels,score_thresh=.2,merge_thresh=.5)
        pred_labels_2.save(output_path,class_config=class_config,crs_transformer=ds.scene.raster_source.crs_transformer)
        # break
      except Exception as e:
          print(f"Skipping {image_id} because of : {e} ")
          continue
          
