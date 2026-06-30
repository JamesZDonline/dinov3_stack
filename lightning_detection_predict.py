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

CONFIG_PATH="object_detection_configs/object_detection_inference_config.yaml"

with open(CONFIG_PATH) as predict_config:
    predict_cfg = yaml.safe_load(predict_config)
    print(yaml.dump(predict_cfg, default_flow_style=False, sort_keys=False))

torch.set_float32_matmul_precision('high')


#Choose which GPU to run on (my code only works with 1 at a time so far)
gpu_id = predict_cfg["hardware_config"]["gpu_id"] 
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
print(f"CUDA_VISIBLE_DEVICES is set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")


#Constants
IMAGERY_BASE_DIRECTORY = predict_cfg["imagery_data"]
LABEL_DIRECTORY = predict_cfg["vector_data"]["label_dir"]
INFERENCE_AOI_DIR = predict_cfg["vector_data"]["inference_aoi_dir"]

batch_size = predict_cfg["data_config"]["data_loader_config"]["batch_size"]
num_workers = predict_cfg["data_config"]["data_loader_config"]["num_workers"]

repo_path = predict_cfg["model_config"]["model_setup"]["repo_path"]
weights_path = predict_cfg["model_config"]["model_setup"]["weights_path"]
model_backbone_name  = predict_cfg["model_config"]["model_setup"]["model_backbone_name"]
lightning_checkpoint_path = predict_cfg["model_config"]["model_setup"].get("lightning_checkpoint_path")

input_channels = predict_cfg["model_config"]["model_setup"]["input_channels"]
clean_weights = predict_cfg["model_config"]["model_setup"]["clean_weights"]

fine_tune = predict_cfg["model_config"]["model_setup"]["fine_tune"]
use_lora = predict_cfg["model_config"]["model_setup"]["use_lora"]
lora_config = predict_cfg["model_config"]["model_setup"]["lora_config"]

patch_dim = predict_cfg["data_config"]["chip_config"]["patch_dim"]
channel_order = predict_cfg["data_config"]["chip_config"]["channel_order"]

upsample_factor = predict_cfg["data_config"]["chip_config"]["upsample_factor"]

learning_rate = predict_cfg["model_config"]["hyperparameters"]["learning_rate"]
smoothing = predict_cfg["model_config"]["hyperparameters"]["label_config"]["smoothing"]
gamma = predict_cfg["model_config"]["hyperparameters"]["label_config"]["gamma"]
early_stop_patience = predict_cfg["model_config"]["hyperparameters"]["early_stop_patience"]

clip_to_aoi = predict_cfg["output_config"]["clip_to_aoi"]
output_dir = predict_cfg["output_config"]["output_dir"]
output_label_suffix = predict_cfg["output_config"]["output_label_suffix"]
stride_factor = predict_cfg["data_config"]["chip_config"]["stride_factor"]


# Setup Class Config
names_list,colors_list = validate_geopacha_class_config(LABEL_DIRECTORY=LABEL_DIRECTORY,run_cfg=predict_cfg)

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

print("COMPILING MODEL FOR INFERENCE...")

# 1. Un-nest the true underlying TorchVisionODAdapter model
pure_torch_model = model.model

# 2. Push the pure model to the GPU and set it to evaluation mode
pure_torch_model = pure_torch_model.to("cuda")
# pure_torch_model.to(torch.bfloat16)

pure_torch_model.eval()
# 3. Compile the pure underlying model with dynamic shape handling enabled
compiled_model = torch.compile(
    pure_torch_model,
    mode="default",
    dynamic=True,
    fullgraph=False
)

pbar = tqdm(os.listdir(INFERENCE_AOI_DIR))
for aoi_path in pbar:
      full_aoi_path = os.path.join(INFERENCE_AOI_DIR,aoi_path)
      aoi = gpd.read_file(full_aoi_path)
      image_id_aoi = aoi['imageid'][0]
      image_id = image_id_aoi.split('_')[0]
      output_path = os.path.join(output_dir,f"{image_id}_{output_label_suffix}")
      pbar.set_description(f"Running prediction on {image_id}")

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
        if not clip_to_aoi: full_aoi_path=f"/mnt/sarl_commons06/Wernke_projects/zimmejr1/All_AOI_10_20_25/ImageID_{image_id}.geojson"
        ds = ObjectDetectionSlidingWindowGeoDataset.from_uris(
              image_uri = full_image_path,
              aoi_uri = full_aoi_path,
              within_aoi=False,
              class_config = class_config,
              size = size,
              stride = int(size*stride_factor),
              out_size = patch_dim,
              image_raster_source_kw=dict(allow_streaming=True,channel_order=channel_order),return_window=False

              )
        ds.scene.id = image_id
        inference_dl = DataLoader(
            ds,
            batch_size=batch_size, 
            shuffle=False,     # Never shuffle during inference
            num_workers=num_workers,     # Parallel loading
            collate_fn=od_collate
        )
        predictions = get_detection_predictions(inference_dl, compiled_model)

        pred_labels = ObjectDetectionLabels.from_predictions(
        ds.windows,
        predictions,
        )
        pred_labels_2 = pred_labels.prune_duplicates(pred_labels,score_thresh=.2,merge_thresh=.5)
        pred_labels_2.save(output_path,class_config=class_config,crs_transformer=ds.scene.raster_source.crs_transformer)
      except Exception as e:
          print(f"Skipping {image_id} because of : {e} ")
          continue
          
