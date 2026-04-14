#Base packages
import os
import pathlib
from tqdm import tqdm
import matplotlib.colors as mcolors
import random

#Data Packages
import pandas as pd
import geopandas as gpd

#Rastervision packages
from rastervision.core.data import ClassConfig, RasterioSource,SemanticSegmentationLabelSourceConfig,SemanticSegmentationLabelSource
from rastervision.pytorch_learner import SemanticSegmentationRandomWindowGeoDataset, SemanticSegmentationSlidingWindowGeoDataset
# from rastervision.pytorch_learner.object_detection_utils import collate_fn as od_collate
# from rastervision.pytorch_learner.object_detection_utils import compute_coco_eval

#Deep Learning Packages
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import draw_segmentation_masks
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassJaccardIndex

# from torchvision.models.detection import retinanet_resnet50_fpn
# from torchvision.ops import sigmoid_focal_loss
# from torchvision.utils import draw_bounding_boxes

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import Callback,TQDMProgressBar
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping



import albumentations as A


#Custom Packages
from src.img_seg.model import Dinov3Segmentation
from geopacha_utilities.utilities import find_pixel_size


import logging
logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)

#Constants
TRAINING_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/AgriculturalSurvey/2026/InitialTest/Training/AOI'
VALIDATION_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/AgriculturalSurvey/2026/InitialTest/Validation/AOI'
# TRAINING_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/AgriculturalSurvey/2026/InitialTest/codetest'
# VALIDATION_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/AgriculturalSurvey/2026/InitialTest/codetest'
LABELS = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/AgriculturalSurvey/2026/InitialTest/labels_4326.geojson'
IMAGERY_BASE_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images'

repo_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3"
weights_path = "/home/VANDERBILT/zimmejr1/Documents/DinoV3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
model_name  = "dinov3_vitl16"

torch.set_float32_matmul_precision('high')

#Variables
patch_dim = 512
batch_size = 16
num_workers = batch_size*2
learning_rate = 5e-5
model_run_name = 'dinov3_ag_seg01'
# smoothing=.01
# gamma = 2
early_stop_patience = 10

AREA_OF_CHIP= (patch_dim*.5)**2


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
        if stage in ("fit",None):
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
            # collate_fn=od_collate,  # Crucial for Raster Vision OD datasets
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=od_collate,
            pin_memory=True
        )

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


# Data Pipeline Configs

data_augmentation_transform = A.Compose([
    A.D4(p=1.0),
])



class_config = ClassConfig(
    names=['Active Broad','Abandoned Broad','Active Sloped','Abandoned Sloped','Active Terrace','Abandoned Terrace','Abandoned Raised','background'],
    colors=['darkgreen','darkred','green','red','pink','lightgreen','purple','black'],
    null_class='background')




import faulthandler
faulthandler.enable()
#Data Pipeline
training_dataset_list = []
val_dataset_list = []
for aoi_path in tqdm(os.listdir(TRAINING_AOI_DIRECTORY),desc="making datasets"):
        full_aoi_path = os.path.join(TRAINING_AOI_DIRECTORY,aoi_path)
        try:
            aoi = gpd.read_file(full_aoi_path)
        except:
            print("failed to read aoi")
        image_id = aoi['imageid'][0]
        # print(image_id)
        # print(image_id)
        image_path  = pathlib.PureWindowsPath(aoi['filepath'][0]).as_posix()
        full_image_path = os.path.join(IMAGERY_BASE_DIRECTORY,image_path)
        full_label_path = LABELS

        #Adjust the patch size to account for different resolution
        rasterSource = RasterioSource(
                full_image_path, #path to the image
                allow_streaming=True, # allow_streaming so we don't have to load the whole image
            ) 
        pixel_size = find_pixel_size(rasterSource.imagery_path)
        size = round(patch_dim*.5/pixel_size)
        num_chips = int(aoi.to_crs('epsg:3857')['geometry'].area/AREA_OF_CHIP)
        max_windows=500


        # print("CREATING DATASET")
        within_aoi=True
        try:
            ds = SemanticSegmentationRandomWindowGeoDataset.from_uris(
                image_uri=full_image_path,
                aoi_uri=full_aoi_path,
                label_vector_uri = full_label_path,
                class_config=class_config,
                image_raster_source_kw=dict(allow_streaming=True,channel_order=[4,2,1]),
                max_windows=max_windows,
                size_lims = [size,size+1],
                out_size=patch_dim,
                within_aoi=within_aoi, 
                transform = data_augmentation_transform
            )
            ds.scene.id=image_id
            training_dataset_list.append(ds)
            # rasterio._loading.get_pipeline()
        except Exception as e: 
                print(f"Couldn't create dataset because:\n{e}")                   
                continue

print("MAKING VALIDATION DATA")
for aoi_path in tqdm(os.listdir(VALIDATION_AOI_DIRECTORY),desc="making datasets"):
        full_aoi_path = os.path.join(VALIDATION_AOI_DIRECTORY,aoi_path)
        aoi = gpd.read_file(full_aoi_path)
        image_id = aoi['imageid'][0]
        # print(image_id)
        image_path  = pathlib.PureWindowsPath(aoi['filepath'][0]).as_posix()
        full_image_path = os.path.join(IMAGERY_BASE_DIRECTORY,image_path)
        full_label_path = LABELS


        #Adjust the patch size to account for different resolution
        rasterSource = RasterioSource(
                full_image_path, #path to the image
                allow_streaming=True, # allow_streaming so we don't have to load the whole image
            ) 
        pixel_size = find_pixel_size(rasterSource.imagery_path)
        size = round(patch_dim*.5/pixel_size)


        try:
              ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
                    image_uri = full_image_path,
                    aoi_uri = full_aoi_path,
                    label_vector_uri = full_label_path,
                    class_config = class_config,
                    size = size,
                    stride = int(size*0.5),
                    out_size = patch_dim,within_aoi=True,
                    image_raster_source_kw=dict(allow_streaming=True,channel_order=[4,2,1]),return_window=False

              )
              ds.scene.id = image_id
              val_dataset_list.append(ds)
        except Exception as e:
              print(f"Problem with {image_id} sliding window: \n {e}")
              continue


dm = ArchaeologyDataModule(
    train_ds_list=training_dataset_list, 
    val_ds_list=val_dataset_list, 
    batch_size=batch_size,
    num_workers=num_workers
)


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


tb_logger = TensorBoardLogger(
    save_dir="logs", 
    name="ag_dinov3_lightning",
    version=model_run_name
)
csv_logger = CSVLogger(save_dir="logs", name="ag_dinov3_lightning")




progress_bar = TQDMProgressBar(refresh_rate=2)
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor="mean_iou",
    dirpath="checkpoints/",
    filename="ag-dinov3-seg-{epoch:02d}-{mean_iou:.3f}",
    save_top_k=1,
    mode="max",
    save_last=True,
)
 
early_stop_callback = EarlyStopping(
    monitor="mean_iou",
    min_delta=0.001,
    patience=early_stop_patience,
    verbose=True,
    mode="max",
)


trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    logger=[tb_logger,csv_logger],  # Connect the logger here
    log_every_n_steps=50,  # How often to log training_loss
    # callbacks = [progress_bar]

    callbacks = [progress_bar,checkpoint_callback, early_stop_callback]
)

# Find batch size

# 3. Start Training
trainer.fit(model, datamodule=dm)


