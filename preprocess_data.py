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
    if os.path.exists(PREPROCESSED_DATA_DIR):
        print(f"Pre-processed data directory '{PREPROCESSED_DATA_DIR}' already exists.")
        print("Delete the directory to re-run preprocessing.")
        return

    print("--- Gathering all data from original source ---")
    all_images_by_id = {}
    all_annotations = []
    master_categories = []

    for dset_name in ["train", "valid", "test"]:
        ann_file_path = os.path.join(ORIGINAL_DATA_DIR, dset_name, "_annotations.coco.json")
        if not os.path.exists(ann_file_path): continue

        with open(ann_file_path, 'r') as f:
            coco_data = json.load(f)

        for img in coco_data['images']:
            img['source_path'] = os.path.join(ORIGINAL_DATA_DIR, dset_name, img['file_name'])
            all_images_by_id[img['id']] = img
        
        all_annotations.extend(coco_data['annotations'])
        if not master_categories and 'categories' in coco_data:
            master_categories = coco_data['categories']

    master_categories.sort(key=lambda x: x['id'])
    category_map = {cat['id']: i + 1 for i, cat in enumerate(master_categories)}

    print(f"Found a total of {len(all_images_by_id)} images and {len(all_annotations)} annotations.")

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
        
        set_images = [all_images_by_id[img_id] for img_id in image_ids]
        set_anns_by_img_id = {img_id: [] for img_id in image_ids}
        for ann in all_annotations:
            if ann['image_id'] in set_anns_by_img_id:
                set_anns_by_img_id[ann['image_id']].append(ann)

        for img_info in tqdm(set_images, desc=f"Processing {set_name}"):
            shutil.copy(img_info['source_path'], os.path.join(output_dir, 'images', img_info['file_name']))
            
            img_anns = set_anns_by_img_id.get(img_info['id'], [])
            if not img_anns: continue

            mask_img = Image.new('L', (img_info['width'], img_info['height']), 0)
            drawer = ImageDraw.Draw(mask_img)
            boxes, labels = [], []

            for i, ann in enumerate(img_anns):
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2)
                    drawer.polygon([tuple(p) for p in poly], fill=i + 1)
                
                bbox = ann['bbox']
                boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                labels.append(category_map[ann['category_id']])

            mask_filename = os.path.splitext(img_info['file_name'])[0] + '.png'
            mask_img.save(os.path.join(output_dir, 'masks', mask_filename))
            
            target_filename = os.path.splitext(img_info['file_name'])[0] + '.npz'
            target_path = os.path.join(output_dir, 'targets', target_filename)
            np.savez(target_path, boxes=np.array(boxes, dtype=np.float32), labels=np.array(labels, dtype=np.int64))

        set_annotations = [ann for ann in all_annotations if ann['image_id'] in image_ids]
        ann_file_path = os.path.join(output_dir, '_annotations.coco.json')
        
        info_block = {"description": f"Banana Ripeness Dataset ({set_name} split)", "year": datetime.date.today().year, "date_created": datetime.datetime.utcnow().isoformat(' ')}
        licenses_block = [{"id": 0, "name": "No License"}]
        
        final_coco = {"info": info_block, "licenses": licenses_block, "images": set_images, "annotations": set_annotations, "categories": master_categories}
        
        with open(ann_file_path, 'w') as f:
            json.dump(final_coco, f)
        print(f"Saved fully compliant annotation file to {ann_file_path}")

    print("\n--- Preprocessing complete! ---")

if __name__ == "__main__":
    run_full_preprocessing()