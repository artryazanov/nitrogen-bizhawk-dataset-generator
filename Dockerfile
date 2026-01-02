# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
# libgl1-mesa-glx and libglib2.0-0 are common requirements for cv2
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY convert_dataset.py .

# Define the entrypoint
ENTRYPOINT ["python", "convert_dataset.py"]

# Default command arguments (can be overridden)
CMD ["--help"]
