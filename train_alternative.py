from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import json
from pathlib import Path
from PIL import Image
import os
import wandb
from datetime import datetime
import logging
from torchvision.ops import box_iou

# Configuration
DATA_ROOT = Path("C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725_Assignment2_2030336/data/auair2019")
COCO_DIR = DATA_ROOT / "coco_annotations"
CHECKPOINT_DIR = Path("./detr_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# AU-AIR classes (must match your COCO conversion)
CLASSES = ["Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer"]
ID2LABEL = {idx+1: cls for idx, cls in enumerate(CLASSES)}
LABEL2ID = {cls: idx+1 for idx, cls in enumerate(CLASSES)}

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize wandb
def init_wandb():
    wandb.init(
        project="au-air-detr",
        name=f"detr-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "architecture": "DETR-ResNet50",
            "dataset": "AU-AIR",
            "classes": CLASSES,
            "img_size": 800,
            "longest_edge": 1333,
        }
    )
    return wandb.config

def get_detr_model(num_classes=8):
    """Initialize DETR model with custom head for AU-AIR classes."""
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = num_classes
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        config=config,
        ignore_mismatched_sizes=True
    )
    return model

def get_feature_extractor():
    """Get DETR image processor with proper normalization."""
    return DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": 800, "longest_edge": 1333}  # Adjust based on your GPU memory
    )

class AUAirDataset(torch.utils.data.Dataset):
    def __init__(self, coco_path, img_dir, feature_extractor):
        try:
            with open(coco_path) as f:
                self.coco_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"COCO annotation file not found: {coco_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from: {coco_path}")
            raise

        self.img_dir = img_dir
        self.feature_extractor = feature_extractor
        self.images = self.coco_data["images"]
        self.annotations = self.coco_data["annotations"]

        # Create image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        anns = self.img_to_anns.get(img_info["id"], [])

        # Format annotations for DETR
        target = {
            "image_id": torch.tensor(img_info["id"]),
            "annotations": []
        }

        for ann in anns:
            target["annotations"].append({
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],  # Keep category_id as is
                "area": ann["area"],
                "iscrowd": ann["iscrowd"]
            })

        # Apply feature extractor
        encoding = self.feature_extractor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        # Remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]  # DETR expects specific target format

        return {
            "pixel_values": pixel_values,
            "labels": target
        }

def collate_fn(batch):
    """Custom collate function for DETR."""
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Stack images and keep labels as list (DETR handles this)
    pixel_values = torch.stack(pixel_values)

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

def compute_metrics(p):
    pred_logits, pred_boxes, labels = p.predictions
    pred_logits = torch.tensor(pred_logits)
    pred_boxes = torch.tensor(pred_boxes)
    labels = torch.tensor(labels)

    # Compute mAP
    map_score = compute_map(pred_logits, pred_boxes, labels)

    return {"eval_map": map_score}

def compute_map(pred_logits, pred_boxes, labels):
    # Compute mAP
    ious = box_iou(pred_boxes, labels[:, :4])
    true_positives = (ious > 0.5).float()
    false_positives = (ious <= 0.5).float()
    precision = true_positives.sum() / (true_positives.sum() + false_positives.sum())
    recall = true_positives.sum() / labels.size(0)
    map_score = (2 * precision * recall) / (precision + recall)
    return map_score.item()

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metrics = {"best_eval_loss": float("inf"), "best_map": 0.0}

    def log(self, logs, start_time=None):
        # Log to wandb
        wandb.log(logs)

        # Track best evaluation loss and mAP
        if "eval_loss" in logs:
            if logs["eval_loss"] < self.best_metrics["best_eval_loss"]:
                self.best_metrics["best_eval_loss"] = logs["eval_loss"]
                logs["best_eval_loss"] = self.best_metrics["best_eval_loss"]

        if "eval_map" in logs:
            if logs["eval_map"] > self.best_metrics["best_map"]:
                self.best_metrics["best_map"] = logs["eval_map"]
                logs["best_map"] = self.best_metrics["best_map"]

        super().log(logs, start_time)

def train_detr():
    # Initialize wandb
    config = init_wandb()

    # Initialize components
    feature_extractor = get_feature_extractor()
    model = get_detr_model()

    # Log model architecture
    wandb.watch(model, log="all")

    # Create datasets
    train_dataset = AUAirDataset(
        COCO_DIR / "train.json",
        DATA_ROOT / "images",
        feature_extractor
    )
    val_dataset = AUAirDataset(
        COCO_DIR / "val.json",
        DATA_ROOT / "images",
        feature_extractor
    )

    # Log dataset statistics
    wandb.log({
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_annotations": len(train_dataset.annotations) + len(val_dataset.annotations)
    })

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=4,  # Adjust based on GPU memory
        per_device_eval_batch_size=2,
        num_train_epochs=50,
        save_steps=500,
        logging_steps=500,  # Reduce logging frequency
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_steps=500,
        gradient_accumulation_steps=2,  # Helps with smaller batch sizes
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        remove_unused_columns=False,  # Important for DETR
        report_to="wandb",  # Enable wandb reporting
        run_name=f"detr-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        fp16=True,  # Enable mixed precision training
        dataloader_num_workers=4,  # Increase the number of workers
        dataloader_pin_memory=True  # Use pinned memory for faster data transfer
    )

    # Initialize Custom Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics  # Add compute_metrics
    )

    # Start training
    trainer.train()

    # Save final model
    model.save_pretrained(str(CHECKPOINT_DIR / "final_model"))
    feature_extractor.save_pretrained(str(CHECKPOINT_DIR / "final_model"))

    # Upload model to wandb
    wandb.save(str(CHECKPOINT_DIR / "final_model" / "*"))

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train_detr()
