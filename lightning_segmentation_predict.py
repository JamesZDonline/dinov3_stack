
from rastervision.core.data import ObjectDetectionLabels
import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import sigmoid_focal_loss
import os
import matplotlib.colors as mcolors
import random
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
import pandas as pd

#Rastervision packages
from rastervision.core.data import ClassConfig, RasterioSource,ObjectDetectionLabelSourceConfig,ObjectDetectionLabelSource


#Deep Learning Packages
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset

from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import sigmoid_focal_loss
from torchvision.utils import draw_bounding_boxes

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import Callback,TQDMProgressBar
from lightning.pytorch.tuner import Tuner
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassJaccardIndex


import albumentations as A


#Custom Packages
from src.img_seg.model import Dinov3Segmentation

from geopacha_utilities.utilities import find_pixel_size, get_segmentation_predictions


#Constants
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

#Constants
# AOI_DIRECTORY = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/OD3/CAA Predict/A'

INFERENCE_AOI = '/home/VANDERBILT/zimmejr1/Documents/GitHub/geopacha_object_detection/OD3/CAA Predict/A'
LABELS = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/AgriculturalSurvey/2026/InitialTest/labels_4326.geojson'
IMAGERY_BASE_DIRECTORY = '/mnt/sda2/Analysis_Images'
OUTPUT_DIR = '/mnt/sda2/AgriculturalSurvey/2026/InitialTest/Deploy'
repo_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3"
# weights_path = "/home/VANDERBILT/zimmejr1/Documents/DinoV3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
model_name  = "dinov3_vitl16"
weights_path = "/home/VANDERBILT/zimmejr1/Documents/DinoV3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"


torch.set_float32_matmul_precision('high')

#Variables
patch_dim = 512
batch_size = 16
num_workers = 0 #batch_size*2
learning_rate = 7.1e-5
model_run_name = 'lr_7e-5_batch_16g2_longer_run'
smoothing=.01
gamma = 2
early_stop_patience = 20

# Model Class


class ArchaeologySegmentationModule(L.LightningModule):
    """
    Lightning wrapper for Dinov3Segmentation.
 
    Batch format from Raster Vision seg datasets:
        images  : FloatTensor  [B, C, H, W]  (already normalised 0-1)
        targets : LongTensor   [B, H, W]     (per-pixel class indices)
    """

    def __init__(self, model, class_config, lr=5e-5,batch_size=8):
        super().__init__()
        self.model = model  
        self.lr = lr
        self.class_config = class_config
        self.num_classes  = len(class_config.names)

        # Weighted cross-entropy handles class imbalance well for archaeology
        # (sparse foreground vs. large background).  Pass a weight tensor here
        # if you want to upweight rare classes further.
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self._train_losses = []

        # Accumulators for epoch-level IoU
        self.val_miou = MulticlassJaccardIndex(
            num_classes=self.num_classes, 
            average=None,          # per-class scores
            ignore_index=255
        )
        self.val_f1 = MulticlassF1Score(num_classes=self.num_classes, average=None)  # average=None gives per-class scores

        # self.validation_step_outputs = []
        # self.training_step_outputs = []
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        logits = self.model(images)

        # Upsample logits to match label resolution if the decoder
        # outputs at a lower stride than the input.
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )

        loss = self.criterion(logits,targets)

        self._train_losses.append(loss.detach())
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False, logger=False)
        return loss
    
    def on_train_epoch_end(self):
        if self._train_losses:
            avg = torch.stack(self._train_losses).mean()
            self.logger.log_metrics({"train_loss": avg}, step=self.current_epoch)
            self._train_losses.clear()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
 
        logits = self.model(images)
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
 
        loss  = self.criterion(logits, targets)
        preds = logits.argmax(dim=1)                       # [B, H, W]
 
        # MeanIoU handles spatial dims natively
        self.val_miou.update(preds, targets)
 
        # MulticlassF1Score needs flat tensors and no ignore_index pixels
        valid_mask      = targets != 255
        preds_flat      = preds[valid_mask]
        targets_flat    = targets[valid_mask]
        self.val_f1.update(preds_flat, targets_flat)
 
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=False)
        return loss

    def on_validation_epoch_end(self):
        iou_per_class = self.val_miou.compute()   # [num_classes]
        f1_per_class  = self.val_f1.compute()     # [num_classes]
 
        self.val_miou.reset()
        self.val_f1.reset()
 
        # Mean IoU and F1 over foreground classes only (exclude background)
        bg_idx  = self.class_config.null_class_id
        fg_mask = torch.ones(self.num_classes, dtype=torch.bool)
        fg_mask[bg_idx] = False
 
        mean_iou     = iou_per_class[fg_mask].mean().item()
        mean_iou_all = iou_per_class.mean().item()
        mean_f1      = f1_per_class[fg_mask].mean().item()
        mean_f1_all  = f1_per_class.mean().item()
 
        metrics = {
            "mean_iou":     mean_iou,
            "mean_iou_all": mean_iou_all,
            "mean_f1":      mean_f1,
            "mean_f1_all":  mean_f1_all,
        }
        for i, name in enumerate(self.class_config.names):
            metrics[f"{name}_iou"] = iou_per_class[i].item()
            metrics[f"{name}_f1"]  = f1_per_class[i].item()
 
        self.logger.log_metrics(metrics, step=self.current_epoch)
        self.log("mean_iou", mean_iou, prog_bar=True, on_epoch=True, sync_dist=True, logger=False)
        self.log("mean_f1",  mean_f1,  prog_bar=True, on_epoch=True, sync_dist=True, logger=False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer



class_config = ClassConfig(
    names=['Active Broad','Abandoned Broad','Active Sloped','Abandoned Sloped','Active Terrace','Abandoned Terrace','Abandoned Raised','background'],
    colors=['darkgreen','darkred','green','red','pink','lightgreen','purple','black'],
    null_class='background')



base_model = Dinov3Segmentation(
    fine_tune=False,
    num_classes=len(class_config.names),
    weights=weights_path,
    model_name=model_name,
    repo_dir=repo_path,
    feature_extractor="multi",
)

model = ArchaeologySegmentationModule(
    model=base_model, 
    class_config=class_config, 
    lr=learning_rate
)

print("Loading Checkpoint")
checkpoint=torch.load("checkpoints/ag-dinov3-seg-epoch=14-mean_iou=0.424.ckpt",map_location="cpu")
model.load_state_dict(checkpoint['state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import ConcatDataset, DataLoader
from rastervision.core.data import SemanticSegmentationLabels
from rastervision.pytorch_learner import SemanticSegmentationRandomWindowGeoDataset, SemanticSegmentationSlidingWindowGeoDataset



# def get_predictions(dataloader,size):
#     for x, _ in tqdm(dataloader):
#         x = x.to(device)
#         with torch.inference_mode():
#             out_batch = model(x)
#             out_batch = out_batch.softmax(dim=1)
#             # Resize to match expected window size (out_size param)
#             if out_batch.shape[-1] != size or out_batch.shape[-2] != size:
#                 out_batch = torch.nn.functional.interpolate(
#                     out_batch, size=(size, size), mode='bilinear', align_corners=True
#                 )
#         for out in out_batch:
#             yield out.cpu().numpy()




# 1. Generate predictions
model.eval()

patch_dim=512

deploy_dataset_list = []
for aoi_path in tqdm(os.listdir(INFERENCE_AOI),desc="making datasets"):
        try:
            full_aoi_path = os.path.join(INFERENCE_AOI,aoi_path)
            aoi = gpd.read_file(full_aoi_path)
            image_id = aoi['imageid'][0]
            image_path  = pathlib.PureWindowsPath(aoi['filepath'][0]).as_posix()
            full_image_path = os.path.join(IMAGERY_BASE_DIRECTORY,image_path)

            #Adjust the patch size to account for different resolution
            rasterSource = RasterioSource(
                    full_image_path, #path to the image
                    allow_streaming=True, # allow_streaming so we don't have to load the whole image

                ) 
            pixel_size = find_pixel_size(rasterSource.imagery_path)
            size = round(patch_dim*.5/pixel_size)

            # print("CREATING DATASET")
            ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
                class_config=class_config,
                image_uri=full_image_path,
                aoi_uri=full_aoi_path,
                # label_vector_uri=LABELS,
                image_raster_source_kw=dict(allow_streaming=True,channel_order=[4,2,1]),
                size=size,
                stride = int(size * 0.75),
                out_size=patch_dim,
                within_aoi=False
            )
            ds.scene.id=image_id
            deploy_dataset_list.append(ds)
        except:
            print(f"{image_id} error.. skipping")
            continue

# import threading
# import queue

# save_queue = queue.Queue(maxsize=4)
# def save_worker():
#     while True:
#         item = save_queue.get()
#         if item is None:  # poison pill to stop worker
#             break
#         pred_labels, output_path, crs_transformer = item
#         print(f"Saving {output_path}")
#         pred_labels.save(
#             uri=output_path,
#             crs_transformer=crs_transformer,
#             smooth_output=False,
#             class_config=class_config,
#             profile_overrides=dict(compress='DEFLATE')
#         )
#         save_queue.task_done()

# # Start save worker thread
# save_thread = threading.Thread(target=save_worker, daemon=True)
# save_thread.start()

for ds in deploy_dataset_list:
    image_id = ds.scene.id
    output_path = os.path.join(OUTPUT_DIR,f"{image_id}_AG_labels.tif")
    print(image_id)
    try:
        if os.path.exists(output_path): continue
        
        inference_dl = DataLoader(
            ds,
            batch_size=16, 
            shuffle=False,     # Never shuffle during inference
            num_workers=32,     # Parallel loading
        )
        predictions = get_segmentation_predictions(dataloader=inference_dl,model=model,size=ds.size[0],device=device)


        pred_labels = SemanticSegmentationLabels.from_predictions(
            ds.windows,
            predictions,
            smooth=False,
            extent=ds.scene.extent,
            num_classes=len(class_config),
            crop_sz=30
        )

        # scores = pred_labels.get_score_arr(pred_labels.extent)

        pred_labels.save(
            uri=OUTPUT_DIR,
            crs_transformer=ds.scene.raster_source.crs_transformer,
            # smooth_output=False,
            class_config=class_config,
            profile_overrides=dict(compress= 'DEFLATE')
        )
        # save_queue.put((pred_labels, output_path, ds.scene.raster_source.crs_transformer))
        os.rename(os.path.join(OUTPUT_DIR,"labels.tif"),output_path)
    except:
        print(f"{image_id} predict error, skipping")
        continue
    
 