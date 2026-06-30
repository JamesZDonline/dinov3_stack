"""
Lightning Object Detection Training Script for Archaeology Projects

This script trains a detection model using PyTorch Lightning for archaeological object detection.
It handles data loading, model setup, training configuration, and experiment tracking.
"""

#Base packages
import os
import faulthandler
import yaml
import logging

#RasterVision packages
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
from geopacha_utilities.model_builder import get_calibrated_dino_model,ArchDetectionModule
from geopacha_utilities.utilities import validate_geopacha_class_config

#Helps with troubleshooting if something fails in the underlying C code
faulthandler.enable()

# Set logging level
logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)

# === CONFIGURATION LOADING ===
CONFIG_PATH="/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3_stack/object_detection_configs/DinoV2_experiments/od_DinoV2_peft_upsample.yaml"

# Load training configuration from YAML file
with open(CONFIG_PATH) as train_config:
    train_cfg = yaml.safe_load(train_config)
    print(yaml.dump(train_cfg, default_flow_style=False, sort_keys=False))

# === HARDWARE SETUP ===
# Select GPU device for training (my code only works with 1 at a time so far)
gpu_id = train_cfg["hardware_config"]["gpu_id"] 
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# Set PyTorch precision for better performance
torch.set_float32_matmul_precision('high')

# === MODEL PARAMETERS ===
# Extract model configuration parameters
model_run_name = train_cfg["experiment_name"]
model_name = train_cfg["model_name"]

# === DATA PATHS ===
TRAINING_AOI_DIRECTORY = train_cfg["vector_data"]["aois"]["training_aoi_dir"]
VALIDATION_AOI_DIRECTORY = train_cfg["vector_data"]["aois"]["validation_aoi_dir"]
LABEL_DIRECTORY = train_cfg["vector_data"]["label_dir"]
IMAGERY_BASE_DIRECTORY = train_cfg["imagery_data"]

# === DATA LOADER CONFIGURATION ===
batch_size = train_cfg["data_config"]["data_loader_config"]["batch_size"]
num_workers = train_cfg["data_config"]["data_loader_config"]["num_workers"]

# === CHIP CONFIGURATION ===
patch_dim = train_cfg["data_config"]["chip_config"]["patch_dim"]
channel_order = train_cfg["data_config"]["chip_config"]["channel_order"]
upsample_factor = train_cfg["data_config"]["chip_config"]["upsample_factor"]
AREA_OF_CHIP = (patch_dim*(.5/upsample_factor))**2

# === MODEL SETUP PARAMETERS ===
repo_path = train_cfg["model_config"]["model_setup"]["repo_path"]
weights_path = train_cfg["model_config"]["model_setup"]["weights_path"]
model_backbone_name = train_cfg["model_config"]["model_setup"]["model_backbone_name"]
input_channels = train_cfg["model_config"]["model_setup"]["input_channels"]
clean_weights = train_cfg["model_config"]["model_setup"]["clean_weights"]

fine_tune = train_cfg["model_config"]["model_setup"]["fine_tune"]
use_lora = train_cfg["model_config"]["model_setup"]["use_lora"]
lora_config = train_cfg["model_config"]["model_setup"]["lora_config"]

lightning_checkpoint_path = train_cfg["model_config"]["model_setup"].get("lightning_checkpoint_path")

# === TRAINING HYPERPARAMETERS ===
learning_rate = train_cfg["model_config"]["hyperparameters"]["learning_rate"]
smoothing = train_cfg["model_config"]["hyperparameters"]["label_config"]["smoothing"]
gamma = train_cfg["model_config"]["hyperparameters"]["label_config"]["gamma"]
early_stop_patience = train_cfg["model_config"]["hyperparameters"]["early_stop_patience"]

# === OUTPUT DIRECTORIES ===
log_dir = train_cfg["output_config"]["log_dir"]
checkpoint_dir = train_cfg["output_config"]["checkpoint_dir"]

# === DATA AUGMENTATION ===
# Define augmentation transforms for training data
data_augmentation_transform = A.Compose([
    A.D4(p=1.0),  # Apply random rotation and flip, random scaling is handeled in the data factory
])

# === CLASS CONFIGURATION ===
# Validate and load class configuration from labels
names_list, colors_list = validate_geopacha_class_config(LABEL_DIRECTORY=LABEL_DIRECTORY, run_cfg=train_cfg)

class_config = ClassConfig(
    names=names_list,
    colors=colors_list,
    null_class='background'
)

# === MODEL CONSTRUCTION ===
# Create the detection model with specified configuration
model = get_calibrated_dino_model(
    num_classes=len(class_config.names),
    weights_path=weights_path,
    model_name=model_backbone_name,
    repo_path=repo_path,
    fine_tune=fine_tune,
    use_lora=use_lora,
    lora_config=lora_config,
    resolution=[patch_dim, patch_dim],
    smoothing=smoothing,
    input_channels=input_channels,
    gamma=gamma,
    clean_weights=clean_weights
)

# Wrap model in TorchVision adapter for compatibility
adapter_model = TorchVisionODAdapter(model)

# Create detection module with Lightning wrapper
model = ArchDetectionModule(
    model=adapter_model, 
    class_config=class_config, 
    lr=learning_rate
)

# === DATA PIPELINE SETUP ===
# Create data factory for handling imagery and labels
data_factory = ObjectDetectionDataFactory(
    image_dir=IMAGERY_BASE_DIRECTORY,
    labels=LABEL_DIRECTORY,
    patch_dim=patch_dim,
    upsample_factor=upsample_factor,
    class_config=class_config,
    augmentation_transform=data_augmentation_transform,
    channel_order=channel_order
)

# Load training and validation datasets
training_dataset_list = data_factory.get_dataset_list(aoi_dir=TRAINING_AOI_DIRECTORY, dtype='training')
val_dataset_list = data_factory.get_dataset_list(aoi_dir=VALIDATION_AOI_DIRECTORY, dtype='validation')

# Create data module for Lightning trainer
dm = ArchaeologyDataModule(
    train_ds_list=training_dataset_list, 
    val_ds_list=val_dataset_list, 
    batch_size=batch_size,
    num_workers=num_workers
)

# === LOGGING SETUP ===
# Configure TensorBoard logger for experiment tracking
tb_logger = TensorBoardLogger(
    save_dir=log_dir, 
    name=model_name,
    version=model_run_name
)

# Configure CSV logger for metrics tracking
csv_logger = CSVLogger(
    save_dir=log_dir, 
    name=model_name,
    version=model_run_name
)

# === CALLBACKS ===
# Progress bar to show training progress
progress_bar = TQDMProgressBar(refresh_rate=2)

# Checkpoint callback to save best models
checkpoint_callback = ModelCheckpoint(
    monitor="mAP50",        # Monitor mean Average Precision at 50 IoU threshold
    dirpath=os.path.join(checkpoint_dir, model_name), 
    filename=model_run_name+"-{epoch:02d}-{mAP50:.3f}",
    save_top_k=1,           # Keep only the single best model
    mode="max",             # We want the highest mAP50
    save_last=True          # Also keep the very latest model just in case
)

# Early stopping callback to prevent overfitting
early_stop_callback = EarlyStopping(
    monitor="mAP50",      # Metric to watch (mean Average Precision)
    min_delta=0.001,      # Minimum change to qualify as an improvement
    patience=early_stop_patience,          # How many epochs to wait without improvement before stopping
    verbose=True,
    mode="max",            # We want to maximize mAP50
    check_on_train_epoch_end=False  # Wait for validation epoch to check
)

# === TRAINING SETUP ===
# Create Lightning trainer with all configurations
trainer = L.Trainer(
    max_epochs=50,          # Maximum number of training epochs
    accelerator="gpu",      # Use GPU acceleration
    devices=1,              # Use single GPU
    precision="16-mixed",   # Use mixed precision training
    logger=[tb_logger, csv_logger],  # Connect the logger here
    log_every_n_steps=50,   # How often to log training_loss
    gradient_clip_val=1.0,  # Gradient clipping to prevent exploding gradients
    callbacks=[progress_bar, checkpoint_callback, early_stop_callback]
)

# === TRAINING EXECUTION ===
# Start the training process with optional checkpoint loading
trainer.fit(model, datamodule=dm, ckpt_path=lightning_checkpoint_path)


