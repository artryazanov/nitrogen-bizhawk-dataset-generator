# 🎮 NitroGen BizHawk Dataset Generator

[![Lua CI](https://github.com/artryazanov/nitrogen-bizhawk-dataset-generator/actions/workflows/lua-ci.yml/badge.svg)](https://github.com/artryazanov/nitrogen-bizhawk-dataset-generator/actions/workflows/lua-ci.yml)
[![Python CI](https://github.com/artryazanov/nitrogen-bizhawk-dataset-generator/actions/workflows/python-ci.yml/badge.svg)](https://github.com/artryazanov/nitrogen-bizhawk-dataset-generator/actions/workflows/python-ci.yml)
[![Docker Build](https://github.com/artryazanov/nitrogen-bizhawk-dataset-generator/actions/workflows/docker-ci.yml/badge.svg)](https://github.com/artryazanov/nitrogen-bizhawk-dataset-generator/actions/workflows/docker-ci.yml)
[![License](https://img.shields.io/github/license/artryazanov/nitrogen-bizhawk-dataset-generator)](LICENSE)
![Lua](https://img.shields.io/badge/Lua-5.4-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

This project provides tools to create training datasets for [NitroGen](https://huggingface.co/nvidia/NitroGen) using the [BizHawk](https://tasvideos.org/BizHawk) emulator.

It consists of two parts:
1.  **Lua Script (`export_dataset.lua`)**: Runs inside BizHawk to export gameplay frames and controller input.
2.  **Python Script (`convert_dataset.py`)**: Converts the exported data into a Parquet file compatible with NitroGen training and pre-processes images.

## 📋 Prerequisites

- **BizHawk Emulator** (Version 2.9+ recommended)
- **Python 3.8+**
- **Git** (optional, for cloning)

## 📦 Installation

1.  Clone this repository or download the files.
2.  Install Python dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Phase 1: 🎞️ Exporting from BizHawk

1.  Open **BizHawk**.
2.  Load your ROM (NES or SNES recommended).
3.  Load a Movie file (`.bk2`) that you want to convert to a dataset.
    *   *Tip: Ensure the movie mode is set to "Play".*
4.  Open the **Lua Console** (`Tools > Lua Console`).
5.  Click **Script > Open Script** and select `export_dataset.lua`.
6.  The script will automatically create a `nitrogen_dataset/` folder and start exporting.
7.  The script will automatically stop when the movie finishes.

> **Note**: The script creates three items in your output directory:
> *   `frames/`: Folder containing raw `frame_XXXXXX.png` images.
> *   `actions.csv`: Raw CSV file with input data.
> *   `dataset_config.json`: Configuration file containing the detected logic (e.g., resize mode based on console).

### Phase 2: 🖼️ Converting and Processing

Once the Lua export is complete, use the Python script to package the data and process the images.

1.  Open a terminal in the project directory.
2.  Run the converter:

```bash
# Default usage 
# Reads from 'nitrogen_dataset/'
# Saves parquet to 'nitrogen_dataset/actions_raw.parquet' (images embedded)
python convert_dataset.py

# Specify custom input directory
python convert_dataset.py --input /path/to/my_export

# Skip image processing (only convert CSV)
python convert_dataset.py --skip-images
```

3.  The output will contain:
    *   `actions_raw.parquet`: The single-file dataset containing both actions and embedded images (ready for training).

### 🐳 Functionality via Docker

You can also run the converter using Docker, which handles all dependencies (including OpenCV) for you.

1.  **Build the Image**:
    ```bash
    docker build -t nitrogen-converter .
    ```

2.  **Run the Container**:
    You need to mount your local dataset folder into the container.
    
    ```bash
    # Run against the 'nitrogen_dataset' folder in your current directory
    docker run --rm -v $(pwd)/nitrogen_dataset:/app/dataset nitrogen-converter --input /app/dataset --output /app/dataset/actions_raw.parquet
    ```


## 🧩 Image Processing Logic

The scripts automatically detect the best resize mode based on the console:

*   **NES**: Uses **Crop** mode (centers and crops to 256x256) to remove overscan borders.
*   **SNES**: Uses **Pad** mode (adds black borders) to maintain aspect ratio within 256x256.

This configuration is saved in `dataset_config.json` by the Lua script and applied by the Python script.

## 🧪 Testing

This project includes tests for both the Python and Lua components.

### 🐍 Python Tests
The Python tests cover image preprocessing and dataset conversion logic.

1.  Calculated dependencies are required (installed via `requirements.txt`), plus `pytest`.
    ```bash
    pip install pytest
    ```
2.  Run the tests:
    ```bash
    pytest tests/
    ```

### 🌙 Lua Tests
The Lua tests validation the input mapping logic and ensure the script structure is correct.

1.  Requires a standard Lua 5.4 interpreter.
2.  Run the tests:
    ```bash
    lua tests/test_export_dataset.lua
    ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
