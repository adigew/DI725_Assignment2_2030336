from pycocotools.coco import COCO

base_path = "C:/Users/nesil.bor/Desktop/Folders/master/DI725/DI725_Assignment2_2030336/data/auair2019/coco_annotations"
for split in ["train", "val", "test"]:
    coco = COCO(f"{base_path}/{split}.json")
    print(f"{split}.json: Images: {len(coco.imgs)}, Annotations: {len(coco.anns)}")
    # Sample first 5 images
    for img_id in coco.getImgIds()[:5]:
        ann_count = len(coco.getAnnIds(imgIds=img_id))
        print(f"  Image ID: {img_id}, Annotations: {ann_count}")