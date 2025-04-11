from pycocotools.coco import COCO

coco = COCO("data/auair2019/coco_annotations/train.json")
print(f"Total Images: {len(coco.imgs)}, Total Annotations: {len(coco.anns)}")

for img_id in coco.getImgIds()[:5]:  # Check first 5 images
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_info = coco.loadImgs(img_id)[0]
    print(f"Image ID: {img_id}, File: {img_info['file_name']}, Annotations: {len(anns)}")
    if anns:
        print(f"Sample Annotation: {anns[0]}")