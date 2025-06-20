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
    Merges all COCO datasets from the original directory, creates a unified category mapping,
    and splits the data into new 'train' and 'valid' sets with consistent annotations.
    """
    if os.path.exists(PREPROCESSED_DATA_DIR):
        print(f"Pre-processed data directory '{PREPROCESSED_DATA_DIR}' already exists.")
        print("Delete the directory to re-run preprocessing.")
        return

    print("--- Step 1: Gathering all data and categories from original source ---")
    all_images_by_id = {}
    all_annotations = []
    all_categories_from_sources = {}

    for dset_name in ["train", "valid", "test"]:
        ann_file_path = os.path.join(ORIGINAL_DATA_DIR, dset_name, "_annotations.coco.json")
        if not os.path.exists(ann_file_path):
            print(f"Warning: Annotation file not found at {ann_file_path}, skipping.")
            continue

        print(f"Reading data from {ann_file_path}")
        with open(ann_file_path, 'r') as f:
            coco_data = json.load(f)

        for img in coco_data.get('images', []):
            img['source_path'] = os.path.join(ORIGINAL_DATA_DIR, dset_name, img['file_name'])
            all_images_by_id[img['id']] = img
        
        all_annotations.extend(coco_data.get('annotations', []))
        
        for cat in coco_data.get('categories', []):
            all_categories_from_sources[cat['id']] = cat
    
    if not all_categories_from_sources:
        raise ValueError("No categories found in any of the source annotation files.")

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

    for ann in all_annotations:
        old_id = ann['category_id']
        ann['category_id'] = old_cat_id_to_new_cat_id[old_id]

    print(f"\nFound a total of {len(all_images_by_id)} images and {len(all_annotations)} annotations.")

    all_image_ids = list(all_images_by_id.keys())
    train_ids, val_ids = train_test_split(all_image_ids, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
    
    data_splits = {
        'train': train_ids,
        'valid': val_ids
    }
    print(f"Created new split: {len(train_ids)} training images, {len(val_ids)} validation images.")

    for set_name, image_ids in data_splits.items():
        print(f"\n--- Processing and saving '{set_name}' set ---")
        
        output_dir = os.path.join(PREPROCESSED_DATA_DIR, set_name)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'targets'), exist_ok=True)
        
        set_images = [img for img_id, img in all_images_by_id.items() if img_id in image_ids]
        set_annotations = [ann for ann in all_annotations if ann['image_id'] in image_ids]

        set_anns_by_img_id = {img_id: [] for img_id in image_ids}
        for ann in set_annotations:
            set_anns_by_img_id[ann['image_id']].append(ann)

        for img_info in tqdm(set_images, desc=f"Processing {set_name}"):
            shutil.copy(img_info['source_path'], os.path.join(output_dir, 'images', img_info['file_name']))
            
            img_anns = set_anns_by_img_id.get(img_info['id'], [])
            if not img_anns:
                continue

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
        
        info_block = {
            "description": f"Saba Dataset - {set_name.capitalize()} Split", 
            "year": datetime.date.today().year,
            "date_created": datetime.datetime.now(datetime.UTC).isoformat()
        }
        licenses_block = [{
            "id": 0, "name": "CC BY 4.0"
        }]
        
        final_coco = {
            "info": info_block, 
            "licenses": licenses_block, 
            "images": set_images, 
            "annotations": set_annotations, 
            "categories": master_categories
        }
        
        with open(ann_file_path, 'w') as f:
            json.dump(final_coco, f, indent=2)
        print(f"Saved fully compliant annotation file to {ann_file_path}")

    print("\n--- Preprocessing complete! ---")

if __name__ == "__main__":
    run_full_preprocessing()