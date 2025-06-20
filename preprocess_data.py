import os
import shutil
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image, ImageDraw
import datetime

# --- CONFIGURATION ---
ORIGINAL_DATA_DIR = "data"
PREPROCESSED_DATA_DIR = "data_preprocessed"
TRAIN_RATIO = 0.85
RANDOM_STATE = 42

def run_full_preprocessing():
    """
    Merges COCO datasets from original train/valid folders, creates a unified category mapping,
    splits the data into new 'train' and 'valid' sets, and processes the original 'test'
    set as a separate holdout set.
    """
    if os.path.exists(PREPROCESSED_DATA_DIR):
        print(f"Pre-processed data directory '{PREPROCESSED_DATA_DIR}' already exists.")
        print("Delete the directory to re-run preprocessing.")
        return

    print("--- Step 1: Gathering all data and creating a unified category map ---")
    
    train_val_images_by_id = {}
    train_val_annotations = []
    test_images_by_id = {}
    test_annotations = []
    all_categories_from_sources = {}

    for dset_name in ["train", "valid"]:
        ann_file_path = os.path.join(ORIGINAL_DATA_DIR, dset_name, "_annotations.coco.json")
        if not os.path.exists(ann_file_path):
            continue
        with open(ann_file_path, 'r') as f:
            coco_data = json.load(f)
        for img in coco_data.get('images', []):
            img['source_path'] = os.path.join(ORIGINAL_DATA_DIR, dset_name, img['file_name'])
            train_val_images_by_id[img['id']] = img
        train_val_annotations.extend(coco_data.get('annotations', []))
        for cat in coco_data.get('categories', []):
            all_categories_from_sources[cat['id']] = cat

    test_ann_file = os.path.join(ORIGINAL_DATA_DIR, "test", "_annotations.coco.json")
    if os.path.exists(test_ann_file):
        with open(test_ann_file, 'r') as f:
            coco_data = json.load(f)
        for img in coco_data.get('images', []):
            img['source_path'] = os.path.join(ORIGINAL_DATA_DIR, "test", img['file_name'])
            test_images_by_id[img['id']] = img
        test_annotations.extend(coco_data.get('annotations', []))
        for cat in coco_data.get('categories', []):
            all_categories_from_sources[cat['id']] = cat
            
    if not all_categories_from_sources:
        raise ValueError("No categories found in any source annotation files.")

    sorted_original_cats = sorted(all_categories_from_sources.values(), key=lambda x: x['name'])
    master_categories = []
    old_cat_id_to_new_cat_id = {}
    for i, cat in enumerate(sorted_original_cats):
        new_id = i + 1
        old_cat_id_to_new_cat_id[cat['id']] = new_id
        new_cat = cat.copy()
        new_cat['id'] = new_id
        master_categories.append(new_cat)

    print("\nUnified Master Category List (New ID -> Name):")
    for cat in master_categories:
        print(f"  {cat['id']}: {cat['name']}")

    for ann in train_val_annotations + test_annotations:
        old_id = ann['category_id']
        if old_id in old_cat_id_to_new_cat_id:
            ann['category_id'] = old_cat_id_to_new_cat_id[old_id]

    def process_and_save_split(set_name, images_by_id, annotations):
        print(f"\n--- Processing and saving '{set_name}' set ---")
        output_dir = os.path.join(PREPROCESSED_DATA_DIR, set_name)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'targets'), exist_ok=True)

        image_ids = list(images_by_id.keys())
        set_images = list(images_by_id.values())
        set_annotations = [ann for ann in annotations if ann['image_id'] in image_ids]
        
        anns_by_img_id = {img_id: [] for img_id in image_ids}
        for ann in set_annotations:
            anns_by_img_id[ann['image_id']].append(ann)

        for img_info in tqdm(set_images, desc=f"Processing {set_name}"):
            shutil.copy(img_info['source_path'], os.path.join(output_dir, 'images', img_info['file_name']))
            img_anns = anns_by_img_id.get(img_info['id'], [])
            if not img_anns: continue

            mask_img = Image.new('L', (img_info['width'], img_info['height']), 0)
            drawer = ImageDraw.Draw(mask_img)
            boxes, labels = [], []

            for i, ann in enumerate(img_anns):
                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape(-1, 2)
                        drawer.polygon([tuple(p) for p in poly], fill=i + 1)
                bbox = ann['bbox']
                boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                labels.append(ann['category_id'])

            mask_filename = os.path.splitext(img_info['file_name'])[0] + '.png'
            mask_img.save(os.path.join(output_dir, 'masks', mask_filename))
            target_filename = os.path.splitext(img_info['file_name'])[0] + '.npz'
            target_path = os.path.join(output_dir, 'targets', target_filename)
            np.savez(target_path, boxes=np.array(boxes, dtype=np.float32), labels=np.array(labels, dtype=np.int64))

        ann_file_path = os.path.join(output_dir, '_annotations.coco.json')
        final_coco = {
            "info": {"description": f"Saba Dataset - {set_name.capitalize()} Split"},
            "images": set_images,
            "annotations": set_annotations,
            "categories": master_categories
        }
        with open(ann_file_path, 'w') as f:
            json.dump(final_coco, f, indent=2)
        print(f"Saved annotation file to {ann_file_path}")

    all_train_val_ids = list(train_val_images_by_id.keys())
    train_ids, val_ids = train_test_split(all_train_val_ids, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
    
    print(f"\nCreated new split: {len(train_ids)} training images, {len(val_ids)} validation images.")
    
    train_images = {k: v for k, v in train_val_images_by_id.items() if k in train_ids}
    valid_images = {k: v for k, v in train_val_images_by_id.items() if k in val_ids}
    
    process_and_save_split('train', train_images, train_val_annotations)
    process_and_save_split('valid', valid_images, train_val_annotations)

    if test_images_by_id:
        process_and_save_split('test', test_images_by_id, test_annotations)

    print("\n--- Full preprocessing complete! ---")

if __name__ == "__main__":
    run_full_preprocessing()