"""
Author: Assistant
Date: 2025-01-17
Version: 1.1

Description:
This script converts prediction file format to match ground truth file format for COCO evaluation.
It handles the conversion of image_id from string format to numeric format by creating a mapping
from the ground truth file, and converts category_id to match the ground truth indexing.

Main functions:
- Creates a mapping between filename and numeric image_id from ground truth
- Converts prediction file image_ids from string to numeric format
- Converts category_ids from 1-based indexing (predictions) to 0-based indexing (ground truth)
- Ensures compatibility with COCO evaluation tools
"""

import json
import os
from pathlib import Path

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save data to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def create_filename_to_id_mapping(gt_file):
    """
    Create mapping from filename to image_id using ground truth file
    
    Args:
        gt_file: Path to ground truth JSON file
        
    Returns:
        dict: Mapping from filename (without extension) to image_id
    """
    gt_data = load_json(gt_file)
    
    filename_to_id = {}
    
    for image_info in gt_data['images']:
        filename = image_info['file_name']
        # Remove file extension to match prediction format
        filename_base = os.path.splitext(filename)[0]
        filename_to_id[filename_base] = image_info['id']
    
    print(f"Created mapping for {len(filename_to_id)} images")
    return filename_to_id

def convert_predictions(pred_file, filename_to_id_mapping, output_file):
    """
    Convert prediction file format to match ground truth
    
    Args:
        pred_file: Path to prediction JSON file
        filename_to_id_mapping: Mapping from filename to numeric image_id
        output_file: Path to save converted predictions
    
    Conversions performed:
        - image_id: string filename -> numeric ID from ground truth mapping
        - category_id: 1-based indexing -> 0-based indexing (subtract 1)
    """
    predictions = load_json(pred_file)
    
    converted_predictions = []
    missing_mappings = set()
    converted_count = 0
    category_id_conversions = {}
    
    for pred in predictions:
        original_image_id = pred['image_id']
        
        # Try to find mapping
        if original_image_id in filename_to_id_mapping:
            # Convert to numeric image_id
            pred['image_id'] = filename_to_id_mapping[original_image_id]
            
            # Convert category_id: predictions start from 1, ground truth starts from 0
            original_category_id = pred['category_id']
            converted_category_id = original_category_id - 1
            pred['category_id'] = converted_category_id
            
            # Track category ID conversions for reporting
            if original_category_id not in category_id_conversions:
                category_id_conversions[original_category_id] = converted_category_id
            
            converted_predictions.append(pred)
            converted_count += 1
        else:
            missing_mappings.add(original_image_id)
    
    # Save converted predictions
    save_json(converted_predictions, output_file)
    
    print(f"Conversion completed:")
    print(f"  - Total original predictions: {len(predictions)}")
    print(f"  - Successfully converted: {converted_count}")
    print(f"  - Missing mappings: {len(missing_mappings)}")
    print(f"  - Category ID conversions: {dict(sorted(category_id_conversions.items()))}")
    
    if missing_mappings:
        print(f"  - Images without mapping (first 10): {list(missing_mappings)[:10]}")
    
    return converted_predictions

def validate_conversion(gt_file, converted_pred_file):
    """
    Validate the converted predictions against ground truth
    
    Args:
        gt_file: Path to ground truth JSON file
        converted_pred_file: Path to converted predictions file
    """
    gt_data = load_json(gt_file)
    pred_data = load_json(converted_pred_file)
    
    # Get image IDs from both files
    gt_image_ids = set(img['id'] for img in gt_data['images'])
    pred_image_ids = set(pred['image_id'] for pred in pred_data)
    
    # Get category IDs
    gt_category_ids = set(cat['id'] for cat in gt_data['categories'])
    pred_category_ids = set(pred['category_id'] for pred in pred_data)
    
    print(f"\nValidation Results:")
    print(f"Ground Truth Images: {len(gt_image_ids)}")
    print(f"Prediction Images: {len(pred_image_ids)}")
    print(f"Common Images: {len(gt_image_ids & pred_image_ids)}")
    print(f"Missing in Predictions: {len(gt_image_ids - pred_image_ids)}")
    print(f"Extra in Predictions: {len(pred_image_ids - gt_image_ids)}")
    
    print(f"\nGround Truth Categories: {sorted(gt_category_ids)}")
    print(f"Prediction Categories: {sorted(pred_category_ids)}")
    print(f"Category Mismatch: {pred_category_ids - gt_category_ids}")

def main():
    """Main function to convert prediction format"""
    
    # File paths - modify these according to your setup
    gt_file = "data/Fisheye/test/test.json"  # Ground truth file
    pred_file = "results/ultralytics/yolov11/x/fisheye_vml3_test/predictions.json"  # Original predictions
    output_file = "results/ultralytics/yolov11/x/fisheye_vml3_test/predictions_converted.json"  # Converted predictions
    
    print("Starting prediction format conversion...")
    
    # Check if files exist
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file not found: {gt_file}")
        return
    
    if not os.path.exists(pred_file):
        print(f"Error: Prediction file not found: {pred_file}")
        return
    
    # Create filename to ID mapping
    print("Creating filename to image_id mapping...")
    filename_to_id = create_filename_to_id_mapping(gt_file)
    
    # Convert predictions
    print("Converting predictions...")
    converted_predictions = convert_predictions(pred_file, filename_to_id, output_file)
    
    # Validate conversion
    print("Validating conversion...")
    validate_conversion(gt_file, output_file)
    
    print(f"\nConversion completed! Converted file saved as: {output_file}")
    print(f"You can now use this file with your COCO.py script.")

if __name__ == "__main__":
    main() 