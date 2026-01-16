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
from rastervision.pytorch_learner.object_detection_utils import compute_coco_eval


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
from geopacha_utilities.utilities import find_pixel_size



os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

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


#Constants
TRAINING_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/training_aoi'
VALIDATION_AOI_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/validation_aoi_sampled'
LABEL_URI = '/mnt/sarl_commons06/Wernke_projects/zimmejr1/DinoV3_Chacu_AOI_12_23_25/chacu_labels.geojson'
IMAGERY_BASE_DIRECTORY = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images'

repo_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3"
weights_path = "/home/VANDERBILT/zimmejr1/Documents/DinoV3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
model_name  = "dinov3_vitl16"

torch.set_float32_matmul_precision('high')

#Variables
patch_dim = 1024
batch_size = 16
num_workers = batch_size*2
learning_rate = 7.1e-5
model_run_name = 'lr_7e-5_batch_16g2_longer_run'
smoothing=.01
gamma = 2
early_stop_patience = 20
# Model Class


def _sum(x):
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res
class CalibratedRetinaNetHead(torch.nn.Module):
    def __init__(self, original_head, smoothing, gamma):
        super().__init__()
        self.original_head = original_head
        self.smoothing = smoothing
        self.gamma = gamma

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = -2

    def forward(self, x):
        return self.original_head(x)

    
    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []

        cls_logits = head_outputs["cls_logits"]

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0

            # --- Apply Label Smoothing ---
            # Formula: target = target * (1 - smoothing) + 0.5 * smoothing
            # This pushes 0.0 to epsilon and 1.0 to 1-epsilon
            if hasattr(self, 'smoothing') and self.smoothing > 0:
                gt_classes_target = gt_classes_target * (1 - self.smoothing) + 0.5 * self.smoothing

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction="sum",
                    gamma=self.gamma
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)
    
    # 2. Function to "Patch" your model for Raster Vision
def get_calibrated_dinov3_model(num_classes, smoothing, gamma):
    # This is where you'd initialize your DINOv3 + RetinaNet model
    model = dinov3_detection(
        fine_tune=True,
        num_classes=2, 
        weights=weights_path,
        model_name=model_name,
        repo_dir=repo_path,
        feature_extractor="multi",
        head="retinanet"
    ) 
    
    # Wrap the existing classification head with our calibrated version
    original_head = model.head.classification_head
    model.head.classification_head = CalibratedRetinaNetHead(
        original_head, 
        smoothing=smoothing, 
        gamma=gamma
    )
    return model

class ChacuDetectionModule(L.LightningModule):
    def __init__(self, model, class_config, lr=5e-5,batch_size=8,num_workers=8):
        super().__init__()
        self.model = model  # model should be the TorchVisionODAdapter(raw_model)
        self.lr = lr
        self.batch_size = batch_size
        self.class_config = class_config
        self.validation_step_outputs = []
        self.training_step_outputs = []
    
    def to_device(self, x, device):
        """Replicating the Raster Vision helper to handle BoxLists/Tensors."""
        if isinstance(x, list):
            return [_x.to(device) if _x is not None else _x for _x in x]
        return x.to(device)
    

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss_dict.values())
        
        # 1. Prepare log dict (detach to save memory)
        log_vars = {k: v.detach().cpu() for k, v in loss_dict.items()}
        log_vars['train_loss'] = total_loss.detach().cpu()
        
        # 2. Append to our list
        self.training_step_outputs.append(log_vars)
        
        # 3. Log to progress bar only (so you see it while training)
        self.log("loss", total_loss, prog_bar=True, on_step=True, on_epoch=False,logger=False)
        
        return total_loss
    
    def forward(self, x):
        # When you call model(x), it will now execute this:
        return self.model(x)
    
    def on_train_epoch_end(self):
        if not self.training_step_outputs:
            return

        # 1. Aggregate and average across the ~47 steps
        keys = self.training_step_outputs[0].keys()
        avg_losses = {}
        for k in keys:
            avg_losses[k] = torch.stack([x[k] for x in self.training_step_outputs]).mean()

        # 2. Log with manual step for RV parity (0, 1, 2...)
        self.logger.log_metrics(avg_losses, step=self.current_epoch)

        # 3. Clear for the next epoch
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        
        # Use the new helper to move everything to CPU for evaluation
        res = {
            'ys': self.to_device(targets, 'cpu'), 
            'outs': self.to_device(outputs, 'cpu')
        }
        
        self.validation_step_outputs.append(res)
        return res

    def on_validation_epoch_end(self):
        # Flatten and compute COCO metrics
        all_ys = []
        all_outs = []
        for out in self.validation_step_outputs:
            all_ys.extend(out['ys'])
            all_outs.extend(out['outs'])

        num_class_ids = len(self.class_config.names)
        coco_eval = compute_coco_eval(all_outs, all_ys, num_class_ids)

        if coco_eval is not None:
            self.logger.log_metrics({
                "mAP50": coco_eval.stats[1],
                "mAP": coco_eval.stats[0]
            }, step=self.current_epoch)
            self.log("mAP50", coco_eval.stats[1], on_epoch=True, prog_bar=True, sync_dist=True,logger=False)
            self.log("mAP", coco_eval.stats[0], on_epoch=True, prog_bar=True,logger=False)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer


model = get_calibrated_dinov3_model(num_classes=2,smoothing=smoothing,gamma=gamma)
from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter
adapter_model = TorchVisionODAdapter(model)

model = ChacuDetectionModule(
    model=adapter_model, 
    class_config=class_config, 
    lr=learning_rate
)
checkpoint=torch.load("checkpoints/chacu-dinov3-epoch=27-mAP50=0.647.ckpt",map_location="cpu")
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
patch_dim=1024
for aoi_path in (os.listdir(AOI_DIRECTORY)):
      print("Making dataset")
      full_aoi_path = os.path.join(AOI_DIRECTORY,aoi_path)
      aoi = gpd.read_file(full_aoi_path)
      image_id_aoi = aoi['imageid'][0]
      image_id = image_id_aoi.split('_')[0]
    #   if image_id !='506412069030':continue
      output_path = os.path.join('/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3_stack/data/train-chacu-round3_lightinging-test/predictions2',f"{image_id}_chacu_labels.geojson")
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
        pred_labels_2 = pred_labels.prune_duplicates(pred_labels,score_thresh=.1,merge_thresh=.1)
        pred_labels_2.save(output_path,class_config=class_config,crs_transformer=ds.scene.raster_source.crs_transformer)
        # break
      except Exception as e:
          print(f"Skipping {image_id} because of : {e} ")
          break
          continue
          
