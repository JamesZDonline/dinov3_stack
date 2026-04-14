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
#Base packages
import os
import pathlib
from tqdm import tqdm

#Data Packages
import geopandas as gpd

#Rastervision packages
from rastervision.core.data import GeoJSONVectorSource, RasterioCRSTransformer, ClassConfig, RasterioSource,ObjectDetectionLabelSourceConfig,ObjectDetectionLabelSource
from rastervision.pytorch_learner import ObjectDetectionRandomWindowGeoDataset,ObjectDetectionSlidingWindowGeoDataset
from rastervision.pytorch_learner.object_detection_utils import collate_fn as od_collate
# from rastervision.pytorch_learner.object_detection_utils import compute_coco_eval


#Deep Learning Packages
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset

from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import sigmoid_focal_loss
from torchvision.utils import draw_bounding_boxes

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import Callback,TQDMProgressBar
from lightning.pytorch.tuner import Tuner


import albumentations as A


#Custom Packages
from src.detection.model import dinov3_detection
from lightning_utils.detection_data import ArchaeologyDataModule, ObjectDetectionDataFactory

from geopacha_utilities.utilities import find_pixel_size

from rastervision.pipeline.file_system import json_to_file, get_tmp_dir
from rastervision.pytorch_learner.object_detection_utils import get_coco_preds, get_coco_gt
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lightning_utils.custom_coco import CustomCOCOeval

import numpy as np

from geopacha_utilities.model_builder import get_calibrated_dinov3_model,ArchDetectionModule



def compute_coco_eval(outputs, targets, num_class_ids):
    """Return mAP averaged over 0.5-0.95 using pycocotools eval.

    Note: boxes are in (ymin, xmin, ymax, xmax) format with values ranging
        from 0 to h or w.

    Args:
        outputs: (list) of length m containing dicts of form
            {'boxes': <tensor with shape (n, 4)>,
             'class_ids': <tensor with shape (n,)>,
             'scores': <tensor with shape (n,)>}
        targets: (list) of length m containing dicts of form
            {'boxes': <tensor with shape (n, 4)>,
             'class_ids': <tensor with shape (n,)>}
    """
    with get_tmp_dir() as tmp_dir:
        preds = get_coco_preds(outputs)
        # ap is undefined when there are no predicted boxes
        if len(preds) == 0:
            return None

        gt = get_coco_gt(targets, num_class_ids)
        gt_path = os.path.join(tmp_dir, 'gt.json')
        json_to_file(gt, gt_path)
        coco_gt = COCO(gt_path)

        pycocotools.coco.unicode = None
        coco_preds = coco_gt.loadRes(preds)

        coco_eval = CustomCOCOeval(cocoGt=coco_gt, cocoDt=coco_preds, iouType='bbox')
        # coco_eval.params.iouThrs=np.linspace(.25, 0.95, int(np.round((0.95 - .25) / .05)) + 1, endpoint=True)



        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# TRAINING_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/training_aoi'
# VALIDATION_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/validation_aoi_sampled'
# LABEL_URI = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/chacu_labels.geojson'
IMAGERY_BASE_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images'
LABEL_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/FirstRun/Labels'

repo_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3"
weights_path = "/home/VANDERBILT/zimmejr1/Documents/DinoV3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
# weights_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3_stack/checkpoints/arch-dinov3-epoch=15-mAP50=0.192.ckpt"

model_name  = "dinov3_vitl16"


import pandas as pd
from tqdm import tqdm
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
VALIDATION_AOI_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/FirstRun/AOI_Validation'
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
model_run_name = 'lr_7e-5_batch_16g2_longer_run'
smoothing=.01
gamma = 2
early_stop_patience = 20
upsample_factor = 2

# Model Class
model = get_calibrated_dinov3_model(num_classes =len(class_config.names), smoothing=smoothing,gamma=gamma)
from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter
adapter_model = TorchVisionODAdapter(model)

model = ArchDetectionModule(
    model=adapter_model, 
    class_config=class_config, 
    lr=learning_rate
)
checkpoint=torch.load("/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3_stack/checkpoints/luna_upsample_plateau2/arch-dinov3-epoch=32-mAP50=0.336.ckpt",map_location="cpu")
model.load_state_dict(checkpoint['state_dict'])


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
model.to("cuda")

# AOI_DIRECTORY = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3-james-experiments/OD_data/Buffered_Chacu_AOI/ExpandedSecondPass"
# datasplit="Training"
# datasplit="Validation"
# AOI_DIRECTORY = f'/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/FirstRun/AOI_{datasplit}'
AOI_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/SecondRunValidate/AOI_Validation'
LABEL_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/SecondRunValidate/Labels'

data_factory = ObjectDetectionDataFactory(image_dir=IMAGERY_BASE_DIRECTORY,labels=LABEL_DIRECTORY,patch_dim=patch_dim,upsample_factor=upsample_factor,
                                          class_config=class_config,augmentation_transform=None)
val_dataset_list = data_factory.get_dataset_list(aoi_dir=AOI_DIRECTORY,dtype='validation',test_code=False)

all_outs = []
all_ys = []

patch_dim=1024
for ds in tqdm(val_dataset_list,desc="Evaluation Run"):
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
            # Collect predictions AND ground truth together
        for x, y in tqdm(inference_dl,desc="evaluate dl",leave=False):
            with torch.inference_mode():
                x_gpu = [img.to('cuda') for img in x]
                outs = model(x_gpu)
            
            all_outs.extend([o.to('cpu') for o in outs])
            all_ys.extend([t.to('cpu') for t in y])

num_class_ids = len(class_config.names)
coco_eval = compute_coco_eval(all_outs, all_ys, num_class_ids)
          
