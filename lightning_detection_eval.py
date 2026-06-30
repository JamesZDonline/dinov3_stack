#Base Packages
import os
import pathlib
import yaml
from tqdm import tqdm

#Data Packages
import geopandas as gpd

#Rastervision packages
from rastervision.core.data import   ClassConfig, ObjectDetectionLabels, RasterioSource
from rastervision.pytorch_learner import ObjectDetectionSlidingWindowGeoDataset
from rastervision.pytorch_learner.object_detection_utils import collate_fn as od_collate
from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter

#Deep Learning Packages
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader

#Custom Packages
from geopacha_utilities.utilities import find_pixel_size, validate_geopacha_class_config, get_detection_predictions
from geopacha_utilities.model_builder import get_calibrated_dino_model, ArchDetectionModule


from rastervision.pytorch_learner import ObjectDetectionLearner
from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import ConcatDataset, DataLoader
from rastervision.core.data import ObjectDetectionLabels
from rastervision.pytorch_learner.object_detection_utils import collate_fn as od_collate

from geopacha_utilities.utilities import compute_coco_eval

from lightning_utils.detection_data import ArchaeologyDataModule, ObjectDetectionDataFactory



CONFIG_PATH="object_detection_configs/object_detection_eval_config.yaml"

with open(CONFIG_PATH) as eval_config:
    eval_cfg = yaml.safe_load(eval_config)
    print(yaml.dump(eval_cfg, default_flow_style=False, sort_keys=False))

torch.set_float32_matmul_precision('high')


#Choose which GPU to run on (my code only works with 1 at a time so far)
gpu_id = eval_cfg["hardware_config"]["gpu_id"] 
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")


#Constants
IMAGERY_BASE_DIRECTORY = eval_cfg["imagery_data"]
LABEL_DIRECTORY = eval_cfg["vector_data"]["label_dir"]
AOI_DIR = eval_cfg["vector_data"]["eval_aoi_dir"]

batch_size = eval_cfg["data_config"]["data_loader_config"]["batch_size"]
num_workers = eval_cfg["data_config"]["data_loader_config"]["num_workers"]

repo_path = eval_cfg["model_config"]["model_setup"]["repo_path"]
weights_path = eval_cfg["model_config"]["model_setup"]["weights_path"]
model_backbone_name  = eval_cfg["model_config"]["model_setup"]["model_backbone_name"]
lightning_checkpoint_path = eval_cfg["model_config"]["model_setup"].get("lightning_checkpoint_path")

input_channels = eval_cfg["model_config"]["model_setup"]["input_channels"]
clean_weights = eval_cfg["model_config"]["model_setup"]["clean_weights"]

fine_tune = eval_cfg["model_config"]["model_setup"]["fine_tune"]
use_lora = eval_cfg["model_config"]["model_setup"]["use_lora"]
lora_config = eval_cfg["model_config"]["model_setup"]["lora_config"]

patch_dim = eval_cfg["data_config"]["chip_config"]["patch_dim"]
channel_order = eval_cfg["data_config"]["chip_config"]["channel_order"]

upsample_factor = eval_cfg["data_config"]["chip_config"]["upsample_factor"]

learning_rate = eval_cfg["model_config"]["hyperparameters"]["learning_rate"]
smoothing = eval_cfg["model_config"]["hyperparameters"]["label_config"]["smoothing"]
gamma = eval_cfg["model_config"]["hyperparameters"]["label_config"]["gamma"]
early_stop_patience = eval_cfg["model_config"]["hyperparameters"]["early_stop_patience"]

# clip_to_aoi = eval_cfg["output_config"]["clip_to_aoi"]
# output_dir = eval_cfg["output_config"]["output_dir"]
# output_label_suffix = eval_cfg["output_config"]["output_label_suffix"]


# Setup Class Config
names_list,colors_list = validate_geopacha_class_config(LABEL_DIRECTORY=LABEL_DIRECTORY,run_cfg=eval_cfg)

class_config = ClassConfig(
    names=names_list,
    colors=colors_list,
    null_class='background')

model = get_calibrated_dino_model(num_classes=len(class_config.names),
                                    weights_path=weights_path,
                                    model_name=model_backbone_name,
                                    repo_path=repo_path,
                                    fine_tune=fine_tune,
                                    use_lora=use_lora,
                                    lora_config=lora_config,
                                    resolution=[patch_dim,patch_dim],
                                    smoothing=smoothing,
                                    input_channels=input_channels,
                                    gamma=gamma,
                                    clean_weights=clean_weights)

adapter_model = TorchVisionODAdapter(model)

model = ArchDetectionModule(
    model=adapter_model, 
    class_config=class_config, 
    lr=learning_rate
)
checkpoint=torch.load(lightning_checkpoint_path,map_location="cpu")
print("LOADING WEIGHTS")
model.load_state_dict(checkpoint['state_dict'],strict=True)
model = model.to("cuda")
model.eval()


print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

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


data_factory = ObjectDetectionDataFactory(image_dir=IMAGERY_BASE_DIRECTORY,labels=LABEL_DIRECTORY,patch_dim=patch_dim,upsample_factor=upsample_factor,
                                          class_config=class_config,augmentation_transform=None,channel_order=channel_order)
val_dataset_list = data_factory.get_dataset_list(aoi_dir=AOI_DIR,dtype='validation',test_code=False)

all_outs = []
all_ys = []

for ds in tqdm(val_dataset_list,desc="Evaluation Run"):

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
                x_gpu = x.to("cuda")
                outs = model.model(x_gpu)
            
            all_outs.extend([o.to('cpu') for o in outs])
            all_ys.extend([t.to('cpu') for t in y])

num_class_ids = len(class_config.names)
coco_eval = compute_coco_eval(all_outs, all_ys, num_class_ids)
          
