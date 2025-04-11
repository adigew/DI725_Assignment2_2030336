import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from transformers import DetrImageProcessor
from detr import get_detr_model
from utils import AUAIRDataset, train_transform, val_transform

def detr_collate_fn(batch):
    """Custom collate function for DETR."""
    return tuple(zip(*batch))

def main(args):
    # Initialize WANDB
    wandb.init(project="DI725_Assignment2_2030336", config={
        "learning_rate": 1e-5,
        "backbone_lr": 1e-4,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    })

    # Config
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_dataset = AUAIRDataset(
        root="C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725_Assignment2_2030336/data/auair2019",
        annotation_file="C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725_Assignment2_2030336/data/auair2019/coco_annotations/train.json",
        transform=train_transform
    )
    val_dataset = AUAIRDataset(
        root="C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725_Assignment2_2030336/data/auair2019",
        annotation_file="C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725_Assignment2_2030336/data/auair2019/coco_annotations/val.json",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=detr_collate_fn
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
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": config.learning_rate},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": config.backbone_lr}
    ]
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = torch.stack(images).to(device).float() / 255.0
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Convert boxes to DETR format [x_center, y_center, w, h]
            detr_targets = []
            for t in targets:
                boxes = t["boxes"]
                if len(boxes) > 0:
                    boxes = boxes.clone()
                    boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
                    boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
                    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
                    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32).to(device)
                detr_targets.append({"boxes": boxes, "class_labels": t["labels"]})
            
            outputs = model(pixel_values=images, labels=detr_targets)
            loss = outputs.loss
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = torch.stack(images).to(device).float() / 255.0
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                detr_targets = []
                for t in targets:
                    boxes = t["boxes"]
                    if len(boxes) > 0:
                        boxes = boxes.clone()
                        boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
                        boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
                        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
                    else:
                        boxes = torch.zeros((0, 4), dtype=torch.float32).to(device)
                    detr_targets.append({"boxes": boxes, "class_labels": t["labels"]})
                outputs = model(pixel_values=images, labels=detr_targets)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        
        # Log to WANDB
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "detr_auair_best.pth")
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), "detr_auair_final.pth")

if __name__ == "__main__":
    import argparse
    from multiprocessing import freeze_support
    
    parser = argparse.ArgumentParser(description="Train DETR on AU-AIR dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()
    
    freeze_support()  # For Windows multiprocessing
    main(args)