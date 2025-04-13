import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from transformers import DetrImageProcessor
import transformers.utils.logging as transformers_logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from detr import get_detr_model
from utils import AUAIRDataset, val_transform

# Suppress Transformers warning
transformers_logging.set_verbosity_error()

def detr_collate_fn(batch):
    """Custom collate function for DETR."""
    return tuple(zip(*batch))

def main(args):
    # Initialize WANDB
    wandb.init(project="DI725_Assignment2_2030336", config={
        "batch_size": args.batch_size,
        "checkpoint": args.checkpoint,
        "score_threshold": args.score_threshold
    })

    # Config
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validation Dataset
    annotation_file = "C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725_Assignment2_2030336/data/auair2019/coco_annotations/val.json"
    val_dataset = AUAIRDataset(
        root="C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725_Assignment2_2030336/data/auair2019",
        annotation_file=annotation_file,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=detr_collate_fn
    )

    # Model
    model = get_detr_model(num_classes=8).to(device)
    try:
        model.load_state_dict(torch.load(config.checkpoint, weights_only=True))
        print(f"Loaded checkpoint from {config.checkpoint}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint {config.checkpoint} not found")

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # COCO Evaluator
    try:
        coco_gt = COCO(annotation_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load COCO annotations from {annotation_file}: {e}")

    # Log categories
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    print("val.json categories:", [{"id": cat["id"], "name": cat["name"]} for cat in categories])

    # Get image IDs
    image_ids = coco_gt.getImgIds()
    if len(image_ids) != len(val_dataset):
        print(f"Warning: Dataset size ({len(val_dataset)}) mismatches COCO image count ({len(image_ids)}).")

    model.eval()
    predictions = []
    batch_idx = 0
    total_valid_preds = 0
    category_counts = {i: 0 for i in range(8)}  # Track predictions per category

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = torch.stack(images).to(device).float() / 255.0
            outputs = model(pixel_values=images)

            # Process predictions
            pred_boxes = outputs.pred_boxes  # [batch, num_queries, 4]
            pred_logits = outputs.logits  # [batch, num_queries, num_classes]

            for i in range(len(images)):
                boxes = pred_boxes[i]
                scores = torch.softmax(pred_logits[i], dim=-1)
                max_scores, labels = scores.max(dim=-1)

                # Log stats
                if batch_idx < 3:
                    print(f"Batch {batch_idx}, Image {i}: Max score = {max_scores.max().item():.4f}, "
                          f"Valid boxes (score > {config.score_threshold}) = {(max_scores > config.score_threshold).sum().item()}, "
                          f"Box ranges: x_min={boxes[:, 0].min().item():.2f}-{boxes[:, 0].max().item():.2f}, "
                          f"w={boxes[:, 2].min().item():.2f}-{boxes[:, 2].max().item():.2f}")

                # Convert boxes to [x_min, y_min, w, h] in pixels
                x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x_min = (x_center - w / 2) * 800
                y_min = (y_center - h / 2) * 800
                w = w * 800
                h = h * 800
                # Clamp to valid ranges
                x_min = torch.clamp(x_min, 0, 800)
                y_min = torch.clamp(y_min, 0, 800)
                w = torch.clamp(w, 0, 800)
                h = torch.clamp(h, 0, 800)
                boxes_pixel = torch.stack([x_min, y_min, w, h], dim=-1)

                # Filter predictions
                valid = (max_scores > config.score_threshold) & (w > 0) & (h > 0) & (~torch.isnan(boxes_pixel).any(dim=-1))
                boxes_pixel = boxes_pixel[valid].cpu().numpy()
                labels = labels[valid].cpu().numpy()
                scores = max_scores[valid].cpu().numpy()

                # Assign image_id
                try:
                    global_idx = batch_idx * config.batch_size + i
                    if global_idx >= len(image_ids):
                        print(f"Warning: Batch index {global_idx} exceeds image IDs ({len(image_ids)}). Skipping.")
                        continue
                    image_id = int(image_ids[global_idx])
                except Exception as e:
                    print(f"Warning: Failed to assign image_id for batch {batch_idx}, image {i}: {e}")
                    continue

                # Prepare COCO format
                for box, label, score in zip(boxes_pixel, labels, scores):
                    if label >= 8:
                        continue
                    pred = {
                        "image_id": image_id,
                        "category_id": int(label),  # Try 0-based first
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "score": float(score)
                    }
                    predictions.append(pred)
                    total_valid_preds += 1
                    category_counts[label] += 1
                    if total_valid_preds <= 10:
                        print(f"Sample prediction {total_valid_preds}: {pred}")

            batch_idx += 1

    # Log results
    if not predictions:
        print("Error: No valid predictions generated. Try lowering score threshold or check model outputs.")
        return
    print(f"Generated {total_valid_preds} valid predictions.")
    print("Predictions per category:", {i: category_counts[i] for i in range(8)})

    # Save predictions
    import json
    with open("predictions.json", "w") as f:
        json.dump(predictions, f)

    # COCO Evaluation
    try:
        coco_dt = coco_gt.loadRes("predictions.json")
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract mAP
        mAP = coco_eval.stats[0]
        mAP_50 = coco_eval.stats[1]
        mAP_75 = coco_eval.stats[2]

        # Log to WANDB
        wandb.log({
            "mAP": mAP,
            "mAP@0.5": mAP_50,
            "mAP@0.75": mAP_75
        })

        print(f"mAP: {mAP:.4f}, mAP@0.5: {mAP_50:.4f}, mAP@0.75: {mAP_75:.4f}")
    except Exception as e:
        print(f"Error during COCO evaluation: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DETR on AU-AIR dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--checkpoint", type=str, default="detr_auair_best.pth", help="Path to model checkpoint")
    parser.add_argument("--score_threshold", type=float, default=0.05, help="Score threshold for predictions")
    args = parser.parse_args()

    main(args)