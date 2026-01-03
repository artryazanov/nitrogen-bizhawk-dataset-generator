import pytest
import numpy as np
import pandas as pd
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import sys

# Add parent directory to path so we can import the script
sys.path.append(str(Path(__file__).parent.parent))

from convert_dataset import preprocess_image, convert_to_nitrogen_format

# --- Image Preprocessing Tests ---

def test_preprocess_image_none():
    assert preprocess_image(None) is None

def test_preprocess_image_stretch():
    # 100x50 image
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    processed = preprocess_image(img, mode="stretch")
    assert processed.shape == (256, 256, 3)

def test_preprocess_image_crop():
    # 100x50 image. Crop should take the center 50x50 and resize to 256x256
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    # Mark the center so we can check if it was preserved (roughly)
    cv2.rectangle(img, (25, 0), (75, 50), (255, 255, 255), -1)
    
    processed = preprocess_image(img, mode="crop")
    assert processed.shape == (256, 256, 3)
    # The center pixel should be white
    assert np.all(processed[128, 128] == [255, 255, 255])

def test_preprocess_image_pad():
    # 100x50 image. Pad should add borders to make it 100x100 then resize.
    img = np.full((50, 100, 3), 255, dtype=np.uint8) # White image
    
    processed = preprocess_image(img, mode="pad")
    assert processed.shape == (256, 256, 3)
    
    # Check that top and bottom are black (padded)
    # 100 width, 50 height -> 25 top pad, 25 bottom pad.
    # Scaled to 256: Ratio is 2.56. 
    # Original image is in the middle 50% height approx.
    # Let's check extreme edges are black
    assert np.all(processed[0, 128] == [0, 0, 0])
    assert np.all(processed[255, 128] == [0, 0, 0])
    # Center should be white
    assert np.all(processed[128, 128] == [255, 255, 255])

def test_preprocess_image_fallback():
    # Test unknown mode defaults to pad
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    processed = preprocess_image(img, mode="unknown_mode")
    assert processed.shape == (256, 256, 3)
    # Should be padded
    assert np.all(processed[0, 0] == [0, 0, 0])

# --- Conversion Logic Tests ---

@pytest.fixture
def temp_dataset_dir(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "frames").mkdir()
    return dataset_dir

def test_convert_to_nitrogen_format_basic(temp_dataset_dir):
    # create dummy actions.csv
    csv_content = "frame,south,east,west,north,start\n1,0,1,0,0,1"
    (temp_dataset_dir / "actions.csv").write_text(csv_content)
    
    output_file = temp_dataset_dir / "output.parquet"
    
    convert_to_nitrogen_format(temp_dataset_dir, output_file, process_images=False)
    
    assert output_file.exists()
    
    df = pd.read_parquet(output_file)
    # Check columns
    expected_bools = ['south', 'east', 'west', 'north', 'left_shoulder', 'right_shoulder', 
                      'left_trigger', 'right_trigger', 'start', 'back', 
                      'dpad_up', 'dpad_down', 'dpad_left', 'dpad_right', 
                      'left_thumb', 'right_thumb', 'guide']
    
    for col in expected_bools:
        assert col in df.columns
        assert df[col].dtype == bool
        
    assert 'j_left' in df.columns
    assert 'j_right' in df.columns
    
    # Check values
    assert df.iloc[0]['east'] == True
    assert df.iloc[0]['start'] == True
    assert df.iloc[0]['south'] == False
    assert df.iloc[0]['left_thumb'] == False # Filled with 0

def test_convert_with_config(temp_dataset_dir):
    # Setup config
    config = {"resize_mode": "crop", "console_type": "NES"}
    (temp_dataset_dir / "dataset_config.json").write_text(json.dumps(config))
    (temp_dataset_dir / "actions.csv").write_text("frame\n1")
    
    # Mock image processing to check if 'crop' mode is passed
    with patch('convert_dataset.preprocess_image') as mock_preprocess:
        # Create a dummy image file
        img_path = temp_dataset_dir / "frames" / "frame_000001.png"
        cv2.imwrite(str(img_path), np.zeros((100,100,3), dtype=np.uint8))
        
        output_file = temp_dataset_dir / "output.parquet"
        convert_to_nitrogen_format(temp_dataset_dir, output_file, process_images=True)
        
        # Verify preprocess_image was called with 'crop'
        args, kwargs = mock_preprocess.call_args
        assert kwargs['mode'] == 'crop'

def test_convert_images_execution(temp_dataset_dir):
    # Validates that images are actually saved
    (temp_dataset_dir / "actions.csv").write_text("frame\n1")
    img_path = temp_dataset_dir / "frames" / "frame_000001.png"
    cv2.imwrite(str(img_path), np.zeros((50, 50, 3), dtype=np.uint8))
    
    output_file = temp_dataset_dir / "output.parquet"
    
    with patch('builtins.print'): # Suppress progress prints
        convert_to_nitrogen_format(temp_dataset_dir, output_file, process_images=True)
    
    # Processed frames directory should NOT exist
    processed_dir = temp_dataset_dir / "processed_frames"
    assert not processed_dir.exists()
    
    # Check parquet file
    assert output_file.exists()
    df = pd.read_parquet(output_file)
    
    assert 'image' in df.columns
    # Check that image data is present and has correct type
    img_bytes = df.iloc[0]['image']
    assert isinstance(img_bytes, bytes)
    
    # Decode image to check it was processed correctly
    decoded_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded_img.shape == (256, 256, 3)
