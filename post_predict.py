from rastervision.core.data import ObjectDetectionLabels
import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import sigmoid_focal_loss
import os
import geopandas as gpd

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

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

TRAINING_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/training_aoi'
VALIDATION_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/validation_aoi_sampled'
LABEL_URI = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/chacu_labels.geojson'
IMAGERY_BASE_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images'
repo_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3"
weights_path = "/home/VANDERBILT/zimmejr1/Documents/DinoV3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
model_name  = "dinov3_vitl16"


class_config = ClassConfig(
    names=['chacu', 'background'],
    colors=['darkred', 'gray'],
    null_class='background')


patch_dim = 1024
model = dinov3_detection(
    fine_tune=True,
    num_classes=2, 
    weights=weights_path,
    model_name=model_name,
    repo_dir=repo_path,
    feature_extractor="multi",
    head="retinanet"
)
from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter
model = TorchVisionODAdapter(model)
model.load_state_dict(torch.load("data/train-chacu-round3/last-model.pth"))



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
            
            # The Adapter returns a list of BoxList objects
            out_batch = model(x)
            
        # Yield each BoxList moved to CPU
        for out in out_batch:
            # Convert the BoxList object to a CPU dictionary
            # Raster Vision's ObjectDetectionLabels expects:
            # {'boxes': np.array, 'class_ids': np.array, 'scores': np.array}
                yield {
                    'boxes': out.convert_boxes('yxyx').cpu().numpy(),
                    'class_ids': out.get_field('class_ids').cpu().numpy(),
                    'scores': out.get_field('scores').cpu().numpy()
                }

# 1. Generate predictions
model.eval()

AOI_DIRECTORY = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3-james-experiments/OD_data/Buffered_Chacu_AOI/ExpandedSecondPass"

for aoi_path in (os.listdir(AOI_DIRECTORY)):
      print("Making dataset")
      full_aoi_path = os.path.join(AOI_DIRECTORY,aoi_path)
      aoi = gpd.read_file(full_aoi_path)
      image_id_aoi = aoi['imageid'][0]
      image_id = image_id_aoi.split('_')[0]
      output_path = os.path.join('/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3_stack/data/train-chacu-round3/predictions',f"{image_id}_chacu_labels.geojson")
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
        size = round(patch_dim*.5/pixel_size)
        ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
              image_uri = full_image_path,
              # aoi_uri = full_aoi_path,
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
        pred_labels_2 = pred_labels.prune_duplicates(pred_labels,score_thresh=.2,merge_thresh=.1)
        pred_labels_2.save(output_path,class_config=class_config,crs_transformer=ds.scene.raster_source.crs_transformer)
      except Exception as e:
          print(f"Skipping {image_id} because of : {e} ")
          continue
          
