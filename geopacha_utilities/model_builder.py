import torch
import torch.nn as nn

from torchvision.ops import sigmoid_focal_loss
from torch.optim import AdamW
import lightning as L


from src.detection.model import dino_detection
from rastervision.pytorch_learner.object_detection_utils import compute_coco_eval

from torchvision.models.detection.image_list import ImageList

class PassThroughTransform(nn.Module):
    def forward(self, images, targets=None):
        # images is already a tensor [B, 8, 896, 896]
        # We just need to wrap it in the ImageList object RetinaNet expects
        image_sizes = [img.shape[-2:] for img in images]
        return ImageList(images, image_sizes), targets
    
    def postprocess(self, result, image_shapes, original_image_sizes):
        # Simply return the results without resizing boxes, masks, or keypoints.
        return result



def _sum(x):
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res

def load_checkpoint_weights(model, weights_path,backbone_has_lora=False,lora_config=None,target_module="backbone.backbone_model",keep_lora_weights=False,strict=False,device="cpu"):
    """
    Load checkpoint weights with support for DDP and LoRA prefixes.
    
    Args:
        model: The model or submodule to load weights into
        checkpoint_path: Path to the checkpoint file
        target_module: Which part of the model to load into (e.g., 'backbone_model'). 
                      If None, loads into the full model.
        skip_lora_adapters: If True, skips lora_A and lora_B weights
        strict: If False, allows missing/unexpected keys
        device: Device to load checkpoint on
        
    Returns:
    """
    print("Load weights")
    checkpoint = torch.load(weights_path,map_location="cpu")
    # 1. Load the checkpoint
    state_dict = checkpoint.get("state_dict", checkpoint)

    # 2. Define the mess we want to clean up
    DDP_PEFT_prefix = "base_model.model.module."
    PEFT_prefix = "base_model.model."

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key
        if  backbone_has_lora:
            new_key = new_key.replace("base_model.model.module.", "base_model.model.")

            if ".base_layer" not in new_key and "lora_" not in new_key:
                for target_mod in model.backbone.lora_config.get("target_modules"):
                    if f".{target_mod}." in new_key:
                        new_key = new_key.replace(f".{target_mod}.", f".{target_mod}.base_layer.")
                        break
            
            # Skip LoRA weights if not keeping them
            if not keep_lora_weights and "lora_" in new_key:
                continue
            
                
        else:
            # Remove the DDP/PEFT prefix
            new_key = new_key.replace(DDP_PEFT_prefix, "")
            new_key = new_key.replace(PEFT_prefix, "")
            new_key = new_key.replace(".base_layer", "")
            
            # Skip the actual LoRA weights (lora_A, lora_B) 
            # unless you have LoRA layers initialized in your current model
            if "lora_" in new_key:
                continue


        cleaned_state_dict[new_key] = value
    
    target = model
    for attr in target_module.split("."):
        target = getattr(target, attr)
    
    msg = target.load_state_dict(cleaned_state_dict, strict=strict)
    print(f"Checkpoint loaded from: {weights_path}")
    print(f"  Backbone has PEFT: {backbone_has_lora}")
    print(f"  Keep LoRA adapters: {keep_lora_weights}")
    print(f"  Missing keys: {len(msg.missing_keys)}")
    print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

    return msg

def get_calibrated_dino_model(
    num_classes,
    weights_path,
    model_name,
    repo_path,
    input_channels=8,
    fine_tune=False,
    use_lora=False,
    lora_config=None,
    resolution=[1024,1024], 
    smoothing=None, 
    gamma=None,
    clean_weights=False):
    """
    Construct and return a calibrated DINO model for object detection.

    This function initializes a DINO-based detection model (either DINOv2 or DINOv3)
    with optional fine-tuning and LoRA support. It also applies calibration to the classification head
    using label smoothing and focal loss gamma parameters.

    Args:
        num_classes (int): Number of object classes for detection.
        weights_path (str): Path to a pre-trained checkpoint to load into the model.
                            If None, no checkpoint is loaded.
        model_name (str): Name of the DINO model variant to use. eg (e.g., 'dinov2_vitl14', 'dinov3_vitl16').
        repo_path (str): Path to the repository containing the model definition.
        input_channels (int, optional): Number of input channels for the model. Default is 8.
        fine_tune (bool, optional): Whether to enable fine-tuning mode. Default is False.
        use_lora (bool, optional): Whether to use LoRA adapters in the backbone. Default is False.
        lora_config (dict, optional): Configuration dictionary for LoRA layers if enabled.
        resolution (list, optional): Input image resolution as [height, width]. Default is [1024, 1024].
        smoothing (float, optional): Label smoothing factor for calibration. Default is None.
        gamma (float, optional): Gamma parameter for focal loss in calibration. Default is None.
        clean_weights (bool, optional): Whether to load weights directly without preprocessing.
                                         Default is False.

    Returns:
        model: A configured RetinaNet-based detection model with label smoothing and focal loss.
    """

    # Check if the model uses DINOv2 architecture
    if 'dinov2' in repo_path:
        clean_weights_path = None

        # If clean_weights flag is set, load weights directly from the path without preprocessing
        if clean_weights and weights_path is not None:
            # Initialize the DINO detection model with specified parameters. If clean_weights is None, default weights will be used.
            model = dino_detection(
                fine_tune=fine_tune,
                use_lora=use_lora,
                weights=clean_weights_path,
                lora_config=lora_config,
                num_classes=num_classes,
                model_name=model_name,
                input_channels=input_channels,
                repo_dir=repo_path,
                resolution=resolution,
                head="retinanet"
            )
        else:
            model = dino_detection(
                fine_tune=fine_tune,
                use_lora=use_lora,
                weights=None,  # Load weights later after cleaning
                lora_config=lora_config,
                num_classes=num_classes,
                model_name=model_name,
                input_channels=input_channels,
                repo_dir=repo_path,
                resolution=resolution,
                head="retinanet"
            )
            if weights_path is not None:
                msg = load_checkpoint_weights(
                    model,
                    weights_path,
                    backbone_has_lora=use_lora,
                    lora_config=lora_config,
                    keep_lora_weights=False)

            print(f"Load Results:\nMissing: {msg.missing_keys}\nUnexpected: {msg.unexpected_keys}")
    else:
        # For DINOv3 models, initialize with standard parameters
        model = dino_detection(
            fine_tune=fine_tune,
            use_lora=use_lora,
            lora_config=lora_config,
            num_classes=num_classes, 
            weights=weights_path,
            model_name=model_name,
            repo_dir=repo_path,
            resolution=resolution,
            feature_extractor="multi",
            head="retinanet"
        ) 
    
    # Replace the default transform with a pass-through transform to avoid unnecessary preprocessing
    model.transform = PassThroughTransform()

    # Wrap the original classification head with a calibrated version that supports smoothing and gamma
    original_head = model.head.classification_head
    model.head.classification_head = CalibratedRetinaNetHead(
        original_head, 
        smoothing=smoothing, 
        gamma=gamma
    )

    return model


class CalibratedRetinaNetHead(torch.nn.Module):
    """
    A calibrated RetinaNet head that extends the original head with label smoothing
    and focal loss adjustments for improved classification performance.

    This module wraps an existing RetinaNet head and modifies its classification
    loss computation to include label smoothing and focal loss with adjustable gamma.

    Args:
        original_head: The original RetinaNet head to be wrapped
        smoothing (float): Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        gamma (float): Focal loss focusing parameter (0.0 = no focus, higher values = more focus on hard examples)
    """
    def __init__(self, original_head, smoothing=0, gamma=0):
        super().__init__()
        self.original_head = original_head
        self.smoothing = smoothing
        self.gamma = gamma

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = -2

    def forward(self, x):
        """
        Forward pass through the original head.

        Args:
            x: Input features from the backbone

        Returns:
            Output from the original head
        """
        return self.original_head(x)

    
    def compute_loss(self, targets, head_outputs, matched_idxs):
        """
        Compute the classification loss with label smoothing and focal loss.

        This method computes the loss for classification targets using sigmoid
        focal loss with optional label smoothing to improve generalization.

        Args:
            targets: List of target dictionaries containing labels and other information
            head_outputs: Dictionary containing classification logits from the head
            matched_idxs: Tensor of matched indices for each anchor

        Returns:
            Total classification loss averaged over all images in the batch
        """
        losses = []

        # Extract classification logits from head outputs
        cls_logits = head_outputs["cls_logits"]

        # Process each image in the batch
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # Identify foreground anchors (those that match with ground truth objects)
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # Create target classification tensor initialized to zeros
            gt_classes_target = torch.zeros_like(cls_logits_per_image)

            # Set the appropriate class labels for foreground anchors
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0

            # Apply label smoothing to prevent overconfidence
            # Formula: target = target * (1 - smoothing) + 0.5 * smoothing
            # This pushes 0.0 to epsilon and 1.0 to 1-epsilon
            if hasattr(self, 'smoothing') and self.smoothing > 0:
                gt_classes_target = gt_classes_target * (1 - self.smoothing) + 0.5 * self.smoothing

            # Identify valid anchors that should not be ignored
            # Anchors with BETWEEN_THRESHOLDS (-2) are ignored in loss computation
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # Compute sigmoid focal loss for foreground anchors
            # The loss is normalized by the number of foreground anchors to prevent
            # the loss from being dominated by background anchors
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction="sum",
                    gamma=self.gamma
                )
                / max(1, num_foreground)
            )

        # Return average loss across all images in the batch
        return _sum(losses) / len(targets)
    

class ArchDetectionModule(L.LightningModule):
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
    
    def on_train_epoch_start(self):
        self.log("learning_rate",self.lr)
        return super().on_train_epoch_start()

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max', patience=5, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "mAP50"},
        }
    
