import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
import os

class AUAIRDataset(Dataset):
    def __init__(self, root, annotation_file, transform=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, "images", img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(ann["category_id"])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        if self.transform:
            transformed = self.transform(image=np.array(img), bboxes=boxes, labels=labels)
            img = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        return img, target

# Define transforms
train_transform = A.Compose([
    A.Resize(800, 800),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format="coco", label_fields=["labels"]))

val_transform = A.Compose([
    A.Resize(800, 800),
    ToTensorV2()
], bbox_params=A.BboxParams(format="coco", label_fields=["labels"]))