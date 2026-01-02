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
    Reads actions.csv, processes it to match NitroGen format, and saves as Parquet.
    Also processes images if configured.
    """
    csv_file = input_dir / "actions.csv"
    config_file = input_dir / "dataset_config.json"
    
    # 1. Load Data
    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        sys.exit(1)
        
    logger.info(f"Reading CSV file: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)

    # 2. Determine Resize Mode
    resize_mode = "pad" # Default
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                resize_mode = config.get("resize_mode", "pad")
                logger.info(f"Loaded config: resize_mode='{resize_mode}' (Console: {config.get('console_type', 'Unknown')})")
        except Exception as e:
            logger.warning(f"Failed to read config file: {e}. Defaulting to 'pad'.")
    else:
        logger.warning("dataset_config.json not found. Defaulting to 'pad'.")

    # 3. Process Images
    if process_images:
        frames_dir = input_dir / "frames"
        processed_dir = input_dir / "processed_frames"
        
        if frames_dir.exists():
            logger.info(f"Processing images from {frames_dir} to {processed_dir} (Mode: {resize_mode})...")
            processed_dir.mkdir(exist_ok=True)
            
            # Iterate through frames
            # Assuming frame_XXXXXX.png naming convention from Lua script
            count = 0
            total = len(list(frames_dir.glob("*.png")))
            
            for img_path in frames_dir.glob("*.png"):
                try:
                    # Read
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Failed to load image: {img_path}")
                        continue
                        
                    # Process
                    processed_img = preprocess_image(img, mode=resize_mode)
                    
                    # Save
                    out_path = processed_dir / img_path.name
                    cv2.imwrite(str(out_path), processed_img)
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count}/{total} images...", end='\r')
                        
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
            print(f"Processed {count}/{total} images. Done.")
        else:
            logger.warning(f"Frames directory not found: {frames_dir}. Skipping image processing.")

    # 4. Process CSV Data (NitroGen Format)
    bool_cols = [
        'south', 'east', 'west', 'north', 
        'left_shoulder', 'right_shoulder', 
        'left_trigger', 'right_trigger', 
        'start', 'back', 
        'dpad_up', 'dpad_down', 'dpad_left', 'dpad_right'
    ]
    extra_required_cols = ['left_thumb', 'right_thumb', 'guide']
    
    for col in bool_cols:
        if col not in df.columns:
            df[col] = 0
    for col in extra_required_cols:
        if col not in df.columns:
            df[col] = 0

    full_bool_list = bool_cols + extra_required_cols
    for col in full_bool_list:
        df[col] = df[col].astype(bool)

    if 'stick_x' in df.columns and 'stick_y' in df.columns:
        df['j_left'] = df.apply(lambda row: [float(row['stick_x']), float(row['stick_y'])], axis=1)
    else:
        df['j_left'] = [[0.0, 0.0]] * len(df)

    df['j_right'] = [[0.0, 0.0]] * len(df)

    final_columns = full_bool_list + ['j_left', 'j_right']
    
    logger.info(f"Saving Parquet to: {output_file}")
    try:
        df[final_columns].to_parquet(output_file, index=False)
        logger.info(f"Successfully converted actions data.")
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
        help="Output Parquet filename (default: <input_dir>/actions_raw.parquet)"
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
        output_file = input_dir / "actions_raw.parquet"
        
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
        
    convert_to_nitrogen_format(input_dir, output_file, process_images=not args.skip_images)

if __name__ == "__main__":
    main()
