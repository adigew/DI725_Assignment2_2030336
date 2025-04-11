import torch
from torch.utils.data import DataLoader
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detr import get_detr_model
from utils import AUAIRDataset, val_transform
import wandb

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "detr_auair_best.pth"
batch_size = 2

# Dataset
test_dataset = AUAIRDataset(
    root="data/au_air",
    annotation_file="data/au_air/coco_annotations/test.json",
    transform=val_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    collate_fn=lambda x: tuple(zip(*x))
)

# Model
model = get_detr_model(num_classes=8)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# COCO evaluation
coco = COCO("data/au_air/coco_annotations/test.json")
results = []
img_id_offset = 0

with torch.no_grad():
    for images, targets in test_loader:
        images = torch.stack(images).to(device)
        outputs = model(pixel_values=images)
        
        for i in range(len(outputs.logits)):
            scores = outputs.logits[i].softmax(-1)[:, :-1]  # Exclude "no object"
            labels = scores.argmax(-1)
            boxes = outputs.pred_boxes[i]  # [x_center, y_center, w, h]
            
            # Convert to COCO [x_min, y_min, w, h]
            boxes_converted = boxes.clone()
            boxes_converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x_min
            boxes_converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y_min
            boxes_converted[:, 2] = boxes[:, 2]  # w
            boxes_converted[:, 3] = boxes[:, 3]  # h
            
            for box, score, label in zip(boxes_converted, scores.max(-1).values, labels):
                if score > 0.5 and label < 8:  # Confidence threshold, exclude "no object"
                    coco_res = {
                        "image_id": test_dataset.ids[img_id_offset + i],
                        "category_id": label.item() + 1,  # COCO IDs start at 1
                        "bbox": box.tolist(),
                        "score": score.item()
                    }
                    results.append(coco_res)
        img_id_offset += len(images)

# Save results
with open("detr_results.json", "w") as f:
    json.dump(results, f)

# Evaluate
coco_dt = coco.loadRes("detr_results.json")
coco_eval = COCOeval(coco, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

mAP = coco_eval.stats[0]  # mAP@0.5
print(f"mAP@0.5: {mAP:.4f}")

# Log to WANDB
wandb.init(project="DI725_Assignment2_STUDENTNUMBER", id="evaluation", resume="allow")
wandb.log({"mAP": mAP})