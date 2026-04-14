#Base packages
import os
import faulthandler
import yaml
import logging

#Rastervision packages
from rastervision.core.data import ClassConfig
from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter

#Deep Learning Packages
import torch
import albumentations as A

#Lightning Packages
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

#Custom Packages
from lightning_utils.detection_data import ArchaeologyDataModule, ObjectDetectionDataFactory
from geopacha_utilities.model_builder import get_calibrated_dinov3_model,ArchDetectionModule
from geopacha_utilities.utilities import validate_geopacha_class_config

#Helps with troubleshooting if something fails in the underlying C code
faulthandler.enable()

logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)

with open("/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3_stack/object_detection_configs/object_detection_train_config.yaml") as train_config:
    train_cfg = yaml.safe_load(train_config)
    print(yaml.dump(train_cfg, default_flow_style=False, sort_keys=False))

#Choose which GPU to run on (my code only works with 1 at a time so far)
gpu_id = train_cfg["hardware_config"]["gpu_id"] 
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

torch.set_float32_matmul_precision('high')


#Import config
model_run_name = train_cfg["experiment_name"]
model_name = train_cfg["model_name"]

TRAINING_AOI_DIRECTORY = train_cfg["vector_data"]["aois"]["training_aoi_dir"]
VALIDATION_AOI_DIRECTORY = train_cfg["vector_data"]["aois"]["validation_aoi_dir"]
LABEL_DIRECTORY = train_cfg["vector_data"]["label_dir"]
IMAGERY_BASE_DIRECTORY = train_cfg["imagery_data"]

batch_size = train_cfg["data_config"]["data_loader_config"]["batch_size"]
num_workers = train_cfg["data_config"]["data_loader_config"]["num_workers"]

patch_dim = train_cfg["data_config"]["chip_config"]["patch_dim"]
upsample_factor = train_cfg["data_config"]["chip_config"]["upsample_factor"]
AREA_OF_CHIP= (patch_dim*(.5/upsample_factor))**2

repo_path = train_cfg["model_config"]["model_setup"]["repo_path"]
weights_path = train_cfg["model_config"]["model_setup"]["weights_path"]
model_name  = train_cfg["model_config"]["model_setup"]["model_name"]
lightning_checkpoint_path = train_cfg["model_config"]["model_setup"].get("lightning_checkpoint_path")


learning_rate = train_cfg["model_config"]["hyperparameters"]["learning_rate"]
smoothing = train_cfg["model_config"]["hyperparameters"]["label_config"]["smoothing"]
gamma = train_cfg["model_config"]["hyperparameters"]["label_config"]["gamma"]
early_stop_patience = train_cfg["model_config"]["hyperparameters"]["early_stop_patience"]


log_dir = train_cfg["output_config"]["log_dir"]
checkpoint_dir = train_cfg["output_config"]["checkpoint_dir"]


# Data Pipeline Configs
data_augmentation_transform = A.Compose([
    A.D4(p=1.0),
])

# Setup Class Config
names_list,colors_list = validate_geopacha_class_config(LABEL_DIRECTORY=LABEL_DIRECTORY,train_cfg=train_cfg)

class_config = ClassConfig(
    names=names_list,
    colors=colors_list,
    null_class='background')


#Data Pipeline
data_factory = ObjectDetectionDataFactory(image_dir=IMAGERY_BASE_DIRECTORY,labels=LABEL_DIRECTORY,patch_dim=patch_dim,upsample_factor=upsample_factor,
                                          class_config=class_config,augmentation_transform=data_augmentation_transform)
training_dataset_list = data_factory.get_dataset_list(aoi_dir=TRAINING_AOI_DIRECTORY,dtype='training')
val_dataset_list = data_factory.get_dataset_list(aoi_dir=VALIDATION_AOI_DIRECTORY,dtype='validation')

dm = ArchaeologyDataModule(
    train_ds_list=training_dataset_list, 
    val_ds_list=val_dataset_list, 
    batch_size=batch_size,
    num_workers=num_workers
)


#Model Setup
model = get_calibrated_dinov3_model(num_classes=len(class_config.names),
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


#Logging Setup
tb_logger = TensorBoardLogger(
    save_dir=log_dir, 
    name=model_name,
    version=model_run_name
)
csv_logger = CSVLogger(
    save_dir=log_dir, 
    name=model_name,
    version=model_run_name
)


progress_bar = TQDMProgressBar(refresh_rate=2)

#Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="mAP50",        # The name we used in self.log
    dirpath=os.path.join(checkpoint_dir,model_name), # Where to save the file
    filename=model_run_name+"-{epoch:02d}-{mAP50:.3f}",
    save_top_k=1,           # Keep only the single best model
    mode="max",             # We want the highest mAP50
    save_last=True          # Also keep the very latest model just in case
)


early_stop_callback = EarlyStopping(
    monitor="mAP50",      # Metric to watch
    min_delta=0.001,      # Minimum change to qualify as an improvement
    patience=early_stop_patience,          # How many epochs to wait without improvement before stopping
    verbose=True,
    mode="max",            # We want to maximize mAP50
    check_on_train_epoch_end=False  # Wait for validation epoch to check
)

#define trainer
trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    logger=[tb_logger,csv_logger],  # Connect the logger here
    log_every_n_steps=50,  # How often to log training_loss
    gradient_clip_val=1.0,  # in Trainer
    callbacks = [progress_bar,checkpoint_callback, early_stop_callback]
)

#train
trainer.fit(model, datamodule=dm, ckpt_path=lightning_checkpoint_path)


