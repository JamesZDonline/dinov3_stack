# model_factory.py content
import os
from rastervision.pytorch_learner.object_detection_utils import TorchVisionODAdapter
from src.detection.model import dinov3_detection

def create_dinov3_model(num_classes: int, **kwargs):
    """
    Factory function to create and wrap the DinoV3 model.
    """
    repo_path = "/home/VANDERBILT/zimmejr1/Documents/GitHub/dinov3"
    weights_path = "/home/VANDERBILT/zimmejr1/Documents/DinoV3_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
    model_name = "dinov3_vitl16"

    # Instantiate the base DinoV3 model
    base_model = dinov3_detection(
        fine_tune=True,
        num_classes=num_classes,
        weights=weights_path,
        model_name=model_name,
        repo_dir=repo_path,
        feature_extractor="multi",
        head="retinanet"
    )

    # Wrap the model for RasterVision compatibility
    return TorchVisionODAdapter(base_model)