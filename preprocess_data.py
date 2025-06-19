import os
import shutil
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
ORIGINAL_DATA_DIR = "data"
PREPROCESSED_DATA_DIR = "data_preprocessed"
TRAIN_RATIO = 0.85
RANDOM_STATE = 42

def gather_all_data():
    all_images = []
    all_annotations = []
    master_categories = []

    print("--- Gathering all data from original source ---")
    # Process train, valid, and test to gather all available data
    for dset_name in ["train", "valid", "test"]:
        ann_file_path = os.path.join(ORIGINAL_DATA_DIR, dset_name, "_annotations.coco.json")
        if not os.path.exists(ann_file_path):
            continue

        with open(ann_file_path, 'r') as f:
            coco_data = json.load(f)

        # Store images with their original source path for later copying
        for img in coco_data['images']:
            img['source_path'] = os.path.join(ORIGINAL_DATA_DIR, dset_name, img['file_name'])
            all_images.append(img)
            
        all_annotations.extend(coco_data['annotations'])

        # Use the first category list found as the master list
        if not master_categories and 'categories' in coco_data:
            master_categories = coco_data['categories']

    # Create a stable mapping from original category ID to a 1-based index
    master_categories.sort(key=lambda x: x['id'])
    category_map = {cat['id']: i + 1 for i, cat in enumerate(master_categories)}

    print(f"Found a total of {len(all_images)} images and {len(all_annotations)} annotations.")
    print(f"Master categories: {[cat['name'] for cat in master_categories]}")
    return all_images, all_annotations, category_map

def create_new_splits(all_images):
    image_ids = [img['id'] for img in all_images]
    
    train_ids, val_ids = train_test_split(image_ids, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
    train_ids, val_ids = set(train_ids), set(val_ids)

    train_imgs = [img for img in all_images if img['id'] in train_ids]
    val_imgs = [img for img in all_images if img['id'] in val_ids]

    print(f"Created new split: {len(train_imgs)} training images, {len(val_imgs)} validation images.")
    return train_imgs, val_imgs

def generate_mask_and_targets(annotations, height, width, category_map):
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    boxes, labels = [], []
    
    # Create an ImageDraw object to draw polygon masks
    mask_img = Image.fromarray(np.zeros((height, width), dtype=np.uint8))
    drawer = ImageDraw.Draw(mask_img)

    for i, ann in enumerate(annotations):
        # Create the instance mask from segmentation polygons
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape(-1, 2)
            # The fill value is the unique instance ID (1-based)
            drawer.polygon([tuple(p) for p in poly], fill=i + 1)
        
        # Extract bounding box and label
        bbox = ann['bbox']
        boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) # to x1,y1,x2,y2
        labels.append(category_map[ann['category_id']])

    # Convert the drawn image back to a numpy array
    combined_mask = np.array(mask_img)
    return combined_mask, np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

def process_and_save_set(set_name, image_list, all_annotations, category_map):
    output_dir = os.path.join(PREPROCESSED_DATA_DIR, set_name)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'targets'), exist_ok=True)

    # Create a lookup dictionary for faster annotation access
    anns_by_img_id = {}
    for ann in all_annotations:
        img_id = ann['image_id']
        if img_id not in anns_by_img_id:
            anns_by_img_id[img_id] = []
        anns_by_img_id[img_id].append(ann)

    print(f"\n--- Processing and saving '{set_name}' set ---")
    for img_info in tqdm(image_list, desc=f"Processing {set_name}"):
        # 1. Copy original image
        if not os.path.exists(img_info['source_path']):
            print(f"Warning: Image file not found {img_info['source_path']}. Skipping.")
            continue
        shutil.copy(img_info['source_path'], os.path.join(output_dir, 'images', img_info['file_name']))
        
        # 2. Get annotations for this image
        img_anns = anns_by_img_id.get(img_info['id'], [])
        if not img_anns:
            continue

        # 3. Generate mask and targets
        mask_array, boxes, labels = generate_mask_and_targets(
            img_anns, img_info['height'], img_info['width'], category_map
        )
        
        # 4. Save mask and targets
        mask_filename = os.path.splitext(img_info['file_name'])[0] + '.png'
        Image.fromarray(mask_array).save(os.path.join(output_dir, 'masks', mask_filename))
        
        target_filename = os.path.splitext(img_info['file_name'])[0] + '.npz'
        target_path = os.path.join(output_dir, 'targets', target_filename)
        np.savez(target_path, boxes=boxes, labels=labels)

def run_full_preprocessing():
    if os.path.exists(PREPROCESSED_DATA_DIR):
        print(f"Pre-processed data directory '{PREPROCESSED_DATA_DIR}' already exists.")
        print("Delete the directory to re-run preprocessing.")
        return

    all_images, all_annotations, category_map = gather_all_data()

    train_imgs, val_imgs = create_new_splits(all_images)
    
    process_and_save_set('train', train_imgs, all_annotations, category_map)
    process_and_save_set('valid', val_imgs, all_annotations, category_map)

    print("\n--- Preprocessing complete! ---")
    print(f"All data has been correctly split and saved to '{PREPROCESSED_DATA_DIR}'.")
    print("You can now proceed with training.")

if __name__ == "__main__":
    run_full_preprocessing()