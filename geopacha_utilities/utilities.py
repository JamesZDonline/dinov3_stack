import rasterio
from rasterio.warp import calculate_default_transform
from pyproj import CRS

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
    

from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import os
def validate_geopacha_class_config(LABEL_DIRECTORY,run_cfg):
    print("Read labels and validate class config")
    label_dataframes=[]
    for label_file in tqdm(os.listdir(LABEL_DIRECTORY)):
        label_dataframes.append(gpd.read_file(os.path.join(LABEL_DIRECTORY,label_file)))
    labels = pd.concat(label_dataframes)

    # Get unique pairs and sort by class_id to ensure order
    mapping_df = labels[['class_id', 'class_name']].drop_duplicates().sort_values('class_id')
    label_names_list = mapping_df['class_name'].tolist()
    label_names_list.append("background")
    ids_list = mapping_df['class_id'].tolist()


    names_list = run_cfg["data_config"]["class_config"]["class_names"]
    if label_names_list != names_list:
        raise RuntimeError("Check config, the names/class_ids in the labels don't match those in the config")
    colors_list = run_cfg["data_config"]["class_config"]["class_colors"]
    if len(colors_list)!=len(names_list):
        raise Warning("Color and class_name lists aren't 1 to 1: generating random colors")
        all_color_names = [c for c in mcolors.CSS4_COLORS.keys() if 'white' not in c and 'snow' not in c]
        colors_list = random.sample(all_color_names, len(names_list))
    return names_list, colors_list

from rastervision.pipeline.file_system import json_to_file, get_tmp_dir
from rastervision.pytorch_learner.object_detection_utils import get_coco_preds, get_coco_gt
import pycocotools
from pycocotools.coco import COCO
from lightning_utils.custom_coco import CustomCOCOeval

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


import torch
def get_detection_predictions(dataloader, model, device='cuda'):
    model.eval()
    model.to(device)
    ds = dataloader.dataset
    native_window_size = ds.size[0]
    model_input_size = ds.out_size[0]

    scale_factor = native_window_size/model_input_size
    for x, _ in tqdm(dataloader):
        with torch.inference_mode():
            # Move images to GPU
            x = [img.to(device) for img in x]
            x = torch.stack(x) 
            
            # The Adapter returns a list of BoxList objects
            out_batch = model(x)
        # Move labels to cpu and yeild them
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


def get_segmentation_predictions(dataloader,model, size, device='cuda'):
    for x, _ in tqdm(dataloader):
        x = x.to(device)
        with torch.inference_mode():
            out_batch = model(x)
            out_batch = out_batch.softmax(dim=1)  # [B, num_classes, H, W]
            if out_batch.shape[-1] != size or out_batch.shape[-2] != size:
                out_batch = torch.nn.functional.interpolate(
                    out_batch,
                    size=(size, size),
                    mode='bilinear',
                    align_corners=False
                )
            out_batch = out_batch.argmax(dim=1)  # [B, H, W] after interpolation
        for out in out_batch:
            yield out.cpu().numpy()