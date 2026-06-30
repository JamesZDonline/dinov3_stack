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
    if 'dinov2' in repo_path:
        clean_weights_path=None
        if clean_weights:
            print("using weights directly")
            clean_weights_path=weights_path
            weights_path=None
        print("Loading and adjusting model")
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
        
        if weights_path is not None:
            msg = load_checkpoint_weights(
                model,
                weights_path,
                backbone_has_lora=use_lora,
                lora_config=lora_config,
                keep_lora_weights=False)

            print(f"Load Results:\nMissing: {msg.missing_keys}\nUnexpected: {msg.unexpected_keys}")
    else:
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
    
    model.transform = PassThroughTransform()

    # Wrap the existing classification head with our calibrated version
    original_head = model.head.classification_head
    model.head.classification_head = CalibratedRetinaNetHead(
        original_head, 
        smoothing=smoothing, 
        gamma=gamma
    )
    return model


class CalibratedRetinaNetHead(torch.nn.Module):
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
    
