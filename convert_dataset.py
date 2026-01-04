import argparse
import logging
import sys
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import pandas as pd
    import pyarrow
    import fastparquet
    import cv2
    import numpy as np
    from datasets import Dataset, Image, Features, Value, Sequence
except ImportError as e:
    print(f"Error: Missing dependency. {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(img: np.ndarray, mode: str = "pad") -> Optional[np.ndarray]:
    """
    Resizes image to 256x256 based on the mode:
    - stretch: simple resize
    - crop: center crop to square, then resize
    - pad: pad with black to square, then resize (default)
    """
    target_size = (256, 256)
    if img is None:
        return None
        
    h, w = img.shape[:2]

    if mode == "stretch":
        if (w, h) != target_size:
            return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img

    elif mode == "crop":
        min_dim = min(h, w)
        if h != w:
            center_h, center_w = h // 2, w // 2
            half_dim = min_dim // 2
            start_h = max(0, center_h - half_dim)
            start_w = max(0, center_w - half_dim)
            end_h = start_h + min_dim
            end_w = start_w + min_dim
            img = img[start_h:end_h, start_w:end_w]
        
        if img.shape[:2] != target_size:
             return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img

    elif mode == "pad":
        max_dim = max(h, w)
        if h != w:
            top = (max_dim - h) // 2
            bottom = max_dim - h - top
            left = (max_dim - w) // 2
            right = max_dim - w - left
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        if img.shape[:2] != target_size:
             return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        return img
    
    # Fallback to pad if unknown mode
    if h != w:
        return preprocess_image(img, "pad")
    if img.shape[:2] != target_size:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img

def convert_to_nitrogen_format(input_dir: Path, output_file: Path, process_images: bool = True) -> None:
    """
    Reads actions.csv, processes images, embeds them into the dataset, and saves as Parquet.
    """
    csv_file = input_dir / "actions.csv"
    config_file = input_dir / "dataset_config.json"
    frames_dir = input_dir / "frames"
    
    # 1. Load CSV Data
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        sys.exit(1)
        
    logger.info(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)

    # 2. Determine Resize Mode
    resize_mode = "pad"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                resize_mode = config.get("resize_mode", "pad")
                logger.info(f"Loaded config: resize_mode='{resize_mode}'")
        except Exception as e:
            logger.warning(f"Failed to read config file: {e}. Defaulting to 'pad'.")

    # 3. Process Images and Embed into DataFrame
    if process_images and frames_dir.exists():
        logger.info(f"Processing images and embedding into Parquet (Mode: {resize_mode})...")
        
        image_bytes_list = []
        total_rows = len(df)
        
        # Iterate through each row of the CSV and look for the corresponding frame
        for index, row in df.iterrows():
            frame_num = int(row['frame'])
            img_path = frames_dir / f"frame_{frame_num:06d}.png"
            
            img_bytes = None
            if img_path.exists():
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Preprocess
                        processed_img = preprocess_image(img, mode=resize_mode)
                        # Encode to PNG bytes
                        success, encoded_img = cv2.imencode('.png', processed_img)
                        if success:
                            img_bytes = encoded_img.tobytes()
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_num}: {e}")
            
            image_bytes_list.append(img_bytes)
            
            if index % 100 == 0:
                print(f"Processed {index}/{total_rows} frames...", end='\r')
        
        # Add column with image bytes
        df['image'] = image_bytes_list
        print(f"Processed {total_rows}/{total_rows} frames. Done.")
    else:
        logger.warning("Skipping image processing or frames directory not found.")

    # 4. Process Actions (NitroGen Format)
    bool_cols = [
        'south', 'east', 'west', 'north', 
        'left_shoulder', 'right_shoulder', 
        'left_trigger', 'right_trigger', 
        'start', 'back', 
        'dpad_up', 'dpad_down', 'dpad_left', 'dpad_right'
    ]
    extra_required_cols = ['left_thumb', 'right_thumb', 'guide']
    
    for col in bool_cols + extra_required_cols:
        if col not in df.columns:
            df[col] = 0
            
    for col in bool_cols + extra_required_cols:
        df[col] = df[col].astype(bool)

    if 'stick_x' in df.columns and 'stick_y' in df.columns:
        df['j_left'] = df.apply(lambda row: [float(row['stick_x']), float(row['stick_y'])], axis=1)
    else:
        df['j_left'] = [[0.0, 0.0]] * len(df)

    df['j_right'] = [[0.0, 0.0]] * len(df)

    # IMPORTANT: Add 'image' to the list of columns to save
    final_columns = bool_cols + extra_required_cols + ['j_left', 'j_right']
    if 'image' in df.columns:
        final_columns.append('image')
    
    logger.info(f"Saving Parquet to: {output_file}")
    
    # 1. Convert Pandas DataFrame to HF Dataset
    ds = Dataset.from_pandas(df[final_columns])
    
    # 2. Cast 'image' column to Image type
    # Hugging Face will automatically understand that it contains PNG bytes
    if 'image' in df.columns:
        ds = ds.cast_column("image", Image())

    # 3. Save using built-in datasets method
    # This creates a Parquet file with the correct schema (Arrow Schema)
    try:
        ds.to_parquet(str(output_file))
        logger.info(f"Successfully converted dataset with HF Image types.")
    except Exception as e:
        logger.error(f"Failed to save Parquet file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert BizHawk CSV export to NitroGen Parquet dataset and process images.")
    
    parser.add_argument(
        "--input", "-i", 
        type=Path, 
        default=Path("nitrogen_dataset"),
        help="Input directory containing actions.csv and frames/ (default: nitrogen_dataset)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=Path, 
        default=None, 
        help="Output Parquet filename (default: <input_dir>/train.parquet)"
    )
    
    parser.add_argument(
        "--skip-images", 
        action="store_true", 
        help="Skip image processing (only convert CSV)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_dir = args.input.resolve()
    
    if args.output:
        output_file = args.output.resolve()
    else:
        output_file = input_dir / "train.parquet"
        
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
        
    convert_to_nitrogen_format(input_dir, output_file, process_images=not args.skip_images)

if __name__ == "__main__":
    main()
