from transformers import DetrForObjectDetection, DetrConfig
import torch

def get_detr_model(num_classes=8):
    """Initialize DETR model for AU-AIR dataset."""
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = num_classes + 1  # 8 classes + "no object"
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        config=config,
        ignore_mismatched_sizes=True
    )
    return model