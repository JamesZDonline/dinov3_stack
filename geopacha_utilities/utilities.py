# Import rasterio and related modules for geospatial data processing
import rasterio
from rasterio.warp import calculate_default_transform
from pyproj import CRS

# Import utilities for progress tracking and data manipulation
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import os

# Import Raster Vision specific modules
from rastervision.pipeline.file_system import json_to_file, get_tmp_dir
from rastervision.pytorch_learner.object_detection_utils import get_coco_preds, get_coco_gt

import torch


# Import COCO evaluation utilities
import pycocotools
from pycocotools.coco import COCO
from lightning_utils.custom_coco import CustomCOCOeval

import random
import matplotlib.colors as mcolors


def find_pixel_size(imagery_path: str) -> float:
    with rasterio.open(imagery_path) as src:
        # 1. Get the center of the image in its native CRS (4326)
        lon = src.lnglat()[0]
        lat = src.lnglat()[1]
        
        # 2. Determine the correct UTM EPSG code automatically
        # UTM zones are 6 degrees wide; 32700 is the prefix for Southern Hemisphere
        zone = int((lon + 180) / 6) + 1
        utm_crs = f"+proj=utm +zone={zone} +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        
        # 3. Calculate transform to that specific UTM zone
        transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds)
        
        # transform[0] is the pixel width in meters
        return abs(transform[0])
    


def validate_geopacha_class_config(LABEL_DIRECTORY, run_cfg):
    """
    Validates that the class configuration in the label data matches the one defined in the run configuration.

    This function reads all GeoJSON files from a specified label directory, combines them into a single
    GeoDataFrame, and then checks whether the unique class names and IDs match those provided in the
    configuration. It also ensures that the number of colors specified matches the number of classes.

    Parameters:
    -----------
    LABEL_DIRECTORY : str
        Path to the directory containing label GeoJSON files.
    run_cfg : dict
        Configuration dictionary containing data_config -> class_config with class_names and class_colors.

    Returns:
    --------
    tuple
        A tuple containing two lists:
        - names_list: List of class names in order of their IDs.
        - colors_list: List of color codes corresponding to each class.

    Raises:
    -------
    RuntimeError
        If the class names or IDs in the labels do not match those in the configuration.
    Warning
        If the number of colors does not match the number of classes, a warning is issued and
        random colors are generated.
    """

    # Read all label files into a list of GeoDataFrames
    label_dataframes = []
    print("Read labels and validate class config")
    for label_file in tqdm(os.listdir(LABEL_DIRECTORY)):
        label_dataframes.append(gpd.read_file(os.path.join(LABEL_DIRECTORY, label_file)))

    # Concatenate all GeoDataFrames into one
    labels = pd.concat(label_dataframes)

    # Extract unique class_id and class_name pairs, sorted by class_id to maintain consistent order
    mapping_df = labels[['class_id', 'class_name']].drop_duplicates().sort_values('class_id')

    # Convert to lists for easier comparison and appending background class
    label_names_list = mapping_df['class_name'].tolist()
    label_names_list.append("background")  # Add background as a special class
    ids_list = mapping_df['class_id'].tolist()

    # Retrieve expected class names from the run configuration
    names_list = run_cfg["data_config"]["class_config"]["class_names"]

    # Validate that labels match configuration
    if label_names_list != names_list:
        raise RuntimeError("Check config: the names/class_ids in the labels don't match those in the config")

    # Retrieve expected colors from the run configuration
    colors_list = run_cfg["data_config"]["class_config"]["class_colors"]

    # Validate that the number of colors matches the number of classes
    if len(colors_list) != len(names_list):


        # Issue warning and generate random colors if mismatch
        print("Warning: Color and class_name lists aren't 1 to 1: generating random colors")
        all_color_names = [c for c in mcolors.CSS4_COLORS.keys() if 'white' not in c and 'snow' not in c]
        colors_list = random.sample(all_color_names, len(names_list))

    return names_list, colors_list


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

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval


def get_detection_predictions(dataloader, model, device='cuda'):
    """
    Generate object detection predictions from a model using the provided dataloader.
    
    This function processes batches of images through the model and yields prediction
    dictionaries containing boxes, class IDs, and scores for each detected object.
    
    Args:
        dataloader: PyTorch DataLoader providing image batches
        model: Trained object detection model
        device (str): Device to run inference on ('cuda' or 'cpu')
        
    Yields:
        dict: Prediction dictionary with keys 'boxes', 'class_ids', and 'scores'
    """
    # Set model to evaluation mode and move to specified device
    model.eval()
    model.to(device)
    
    # Get dataset properties for scaling boxes
    ds = dataloader.dataset
    native_window_size = ds.size[0]
    model_input_size = ds.out_size[0]
    scale_factor = native_window_size / model_input_size
    
    # Process each batch in the dataloader
    for x, _ in tqdm(dataloader):
        with torch.inference_mode():
            # Move images to GPU
            x = [img.to(device) for img in x]
            x = torch.stack(x) 
            
            # The Adapter returns a list of BoxList objects
            out_batch = model(x)
        # Move labels to cpu and yield them
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


def get_segmentation_predictions(dataloader, model, size, device='cuda'):
    """
    Generate segmentation predictions from a model using the provided dataloader.
    
    This function processes batches of images through the model and yields 
    segmentation masks for each image in the batch.
    
    Args:
        dataloader: PyTorch DataLoader providing image batches
        model: Trained segmentation model
        size (int): Target output size for segmentation masks
        device (str): Device to run inference on ('cuda' or 'cpu')
        
    Yields:
        numpy.ndarray: Segmentation mask with shape [H, W] 
    """
    # Process each batch in the dataloader
    for x, _ in tqdm(dataloader):
        # Move input to specified device
        x = x.to(device)
        
        with torch.inference_mode():
            # Run model inference
            out_batch = model(x)
            
            # Apply softmax to get class probabilities
            out_batch = out_batch.softmax(dim=1)  # [B, num_classes, H, W]
            
            # Resize output to target size if needed
            if out_batch.shape[-1] != size or out_batch.shape[-2] != size:
                out_batch = torch.nn.functional.interpolate(
                    out_batch,
                    size=(size, size),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Convert probabilities to class predictions (argmax)
            out_batch = out_batch.argmax(dim=1)  # [B, H, W] after interpolation
        
        # Yield each prediction mask in the batch
        for out in out_batch:
            yield out.cpu().numpy()