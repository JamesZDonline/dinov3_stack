import torch
import torch.nn as nn
import os
from peft import get_peft_model, LoraConfig, TaskType

from torchvision.models.detection.ssd import (
    SSD, 
    DefaultBoxGenerator,
    SSDHead
)
from torchvision.models.detection.retinanet import (
    RetinaNet, RetinaNetHead, AnchorGenerator
)

def load_model(weights: str=None, model_name: str=None, repo_dir: str=None):
    if weights is not None:
        print('Loading pretrained backbone weights from: ', weights)
        model = torch.hub.load(
            repo_dir, 
            model_name, 
            source='local', 
            weights=weights
        )
    else:
        print('No pretrained weights path given. Loading with random weights.')
        model = torch.hub.load(
            repo_dir, 
            model_name, 
            source='local'
        )
    
    return model

class DinoBackbone(nn.Module):
    def __init__(self, 
        weights: str=None,
        model_name: str=None,
        repo_dir: str=None,
        fine_tune: bool=False,
        use_lora: bool=False,
        lora_config: dict=None,
        input_channels: int=3

    ):
        super(DinoBackbone, self).__init__()

        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_config = {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.1,
                    "target_modules": ["qkv", "proj"]
                }

        self.backbone_model = load_model(
            weights=weights, model_name=model_name, repo_dir=repo_dir
        )

        if use_lora and not fine_tune:
            print("Warning: use_lora=True but fine_tune=False. LoRA will be ignored.")
        
        if input_channels!=3:
            old_proj = self.backbone_model.patch_embed.proj

            new_proj = nn.Conv2d(
                in_channels=input_channels,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding
                ).requires_grad_(False)

            self.backbone_model.patch_embed.proj = new_proj


        for name, param in self.backbone_model.named_parameters():
                param.requires_grad = False
        
        if fine_tune:
            if use_lora:
                if lora_config is None:
                    lora_config = {}
                
                # Set defaults
                self.lora_config.update(lora_config)
                peft_config = LoraConfig(
                    r=self.lora_config["r"],
                    lora_alpha=self.lora_config["lora_alpha"],
                    lora_dropout=self.lora_config["lora_dropout"],
                    bias="none",
                    target_modules=self.lora_config["target_modules"],
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                self.backbone_model = get_peft_model(self.backbone_model, peft_config)
            else:
                for name, param in self.backbone_model.named_parameters():
                    param.requires_grad = True
       

    def forward(self, x):
        out = self.backbone_model.get_intermediate_layers(
            x, 
            n=1, 
            reshape=True, 
            return_class_token=False, 
            norm=True
        )[0]

        return out

def dino_detection(
    fine_tune: bool=False,
    use_lora: bool=False,
    lora_config: dict=None,
    num_classes: int=2,
    weights: str=None,
    model_name: str=None,
    repo_dir: str=None,
    resolution: list=[640, 640],
    input_channels: int=3,
    nms: float=0.45,
    feature_extractor: str='last', # OR 'multi'
    head: str='ssd' # Detection head type, ssd or retinanet
):
    backbone = DinoBackbone(
        weights=weights, 
        model_name=model_name, 
        repo_dir=repo_dir, 
        fine_tune=fine_tune,
        input_channels=input_channels,
        use_lora=use_lora,
        lora_config=lora_config
    )

    if head == 'ssd':
        out_channels = [backbone.backbone_model.norm.normalized_shape[0]] * 6
        anchor_generator = DefaultBoxGenerator(
            aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        )
        
        num_anchors = anchor_generator.num_anchors_per_location()
        det_head = SSDHead(out_channels, num_anchors, num_classes=num_classes)
    
        model = SSD(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            size=resolution,
            head=det_head,
            nms_thresh=nms
        )
    
    elif head == 'retinanet':
        backbone.out_channels = backbone.backbone_model.norm.normalized_shape[0]
        anchor_sizes = ((32, 64, 128, 256, 512),)  # one tuple, for one feature map
        aspect_ratios = ((0.5, 1.0, 2.0),)         # one tuple, same idea
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        model = RetinaNet(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            min_size=resolution[0],
            max_size=resolution[1],
            # head=head,
            nms_thresh=nms
        )

    return model


if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    from src.utils.common import get_dinov3_paths
    from torchinfo import summary

    import numpy as np
    import os

    DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()

    input_size = 640

    transform = transforms.Compose([
        transforms.Resize(
            input_size, 
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )
    ])

    model_names = {
        'dinov3_convnext_tiny': 'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
        'dinov3_convnext_small': 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
        'dinov3_convnext_base': 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
        'dinov3_convnext_large': 'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
        'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vits16plus': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
        'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'dinov3_vith16plus': 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
    }

    for head in ['ssd', 'retinanet']:
        print(f"Building {head} models...\n\n")
        for model_name in model_names:
            print('Testing: ', model_name)
            # model = DinoDetection(
            #     repo_dir=DINOV3_REPO, 
            #     weights=os.path.join(DINOV3_WEIGHTS, model_names[model_name]),
            #     model_name=model_name,
            #     feature_extractor='last' # OR 'last'
            # )
            model = dino_detection(
                repo_dir=DINOV3_REPO, 
                weights=os.path.join(DINOV3_WEIGHTS, model_names[model_name]),
                model_name=model_name,
                feature_extractor='last', # OR 'last'
                head=head
            )
            model.eval()
            print(model)
        
            random_image = Image.fromarray(np.ones(
                (input_size, input_size, 3), dtype=np.uint8)
            )
            x = transform(random_image).unsqueeze(0)
        
            with torch.no_grad():
                outputs = model(x)
            
            print(outputs)
        
            summary(
                model, 
                input_data=x,
                col_names=('input_size', 'output_size', 'num_params'),
                row_settings=['var_names'],
            )
            print('#' * 50, '\n\n')