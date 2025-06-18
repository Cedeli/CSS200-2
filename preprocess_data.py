import os
import shutil
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
from tqdm import tqdm

def run_preprocessing():
    """
    This script converts the raw COCO dataset into an optimized format for faster training.
    It generates combined masks and saves targets (boxes, labels) as numpy files.
    """
    DATASETS = [
        {
            "name": "train",
            "img_dir": "data/train",
            "ann_file": "data/train/_annotations.coco.json"
        },
        {
            "name": "valid",
            "img_dir": "data/valid",
            "ann_file": "data/valid/_annotations.coco.json"
        },
        {
            "name": "test",
            "img_dir": "data/test",
            "ann_file": "data/test/_annotations.coco.json"
        }
    ]
    PREPROCESSED_DATA_DIR = "data_preprocessed"

    if os.path.exists(PREPROCESSED_DATA_DIR):
        print(f"Pre-processed data directory '{PREPROCESSED_DATA_DIR}' already exists.")

    print("Starting data pre-processing...")
    for dataset_info in DATASETS:
        output_dir = os.path.join(PREPROCESSED_DATA_DIR, dataset_info["name"])
        preprocess_single_dataset(dataset_info["img_dir"], dataset_info["ann_file"], output_dir)
    print("All datasets have been pre-processed successfully!")

def preprocess_single_dataset(img_dir, ann_file, output_dir):
    """Processes a single dataset split."""
    # Create output subdirectories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'targets'), exist_ok=True)

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    
    coco_cat_ids = sorted(coco.getCatIds())
    category_map = {cat_id: i + 1 for i, cat_id in enumerate(coco_cat_ids)}

    print(f"Processing {len(img_ids)} images from '{img_dir}'...")

    for img_id in tqdm(img_ids, desc=f"Processing {os.path.basename(img_dir)}"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found {img_path}. Skipping.")
            continue
            
        shutil.copy(img_path, os.path.join(output_dir, 'images', img_info['file_name']))
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if not anns:
            continue

        combined_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        boxes, labels = [], []

        for i, ann in enumerate(anns):
            instance_mask = coco.annToMask(ann)
            combined_mask[(instance_mask == 1) & (combined_mask == 0)] = i + 1 
            
            bbox = ann['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(category_map[ann['category_id']])

        mask_filename = os.path.splitext(img_info['file_name'])[0] + '.png'
        Image.fromarray(combined_mask).save(os.path.join(output_dir, 'masks', mask_filename))
        
        target_filename = os.path.splitext(img_info['file_name'])[0] + '.npz'
        target_path = os.path.join(output_dir, 'targets', target_filename)
        np.savez(target_path, boxes=np.array(boxes, dtype=np.float32), labels=np.array(labels, dtype=np.int64))

if __name__ == "__main__":
    run_preprocessing()