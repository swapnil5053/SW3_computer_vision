#!/bin/bash

# ARD-CNN Raindrop Removal Script
# This script runs the ARD-CNN model on images in the dataset folder

BASEDIR="$(dirname $0)"
PROJECT_ROOT="$BASEDIR/.."

echo "ARD-CNN Raindrop Removal"
echo "========================"

# Check if Docker is available and image exists
if command -v docker &> /dev/null; then
    if docker images | grep -q "ardcnn"; then
        echo "Using Docker container..."
        
        # Run with Docker (GPU support)
        docker run -it --gpus all \
            -v "$(pwd)/../dataset:/workspace/dataset" \
            -v "$(pwd)/../model:/workspace/model" \
            -v "$(pwd)/../output:/workspace/output" \
            -v "$(pwd)/../removal:/workspace/removal" \
            -w /workspace/removal \
            ardcnn:cuda12 \
            python test_ardcnn.py
    else
        echo "Docker image 'ardcnn:cuda12' not found."
        echo "Please build the Docker image first using: ./build-cuda.sh"
        exit 1
    fi
else
    echo "Docker not found. Running locally..."
    
    # Check if we're in the right directory
    if [ ! -f "$PROJECT_ROOT/removal/test_ardcnn.py" ]; then
        echo "Error: test_ardcnn.py not found. Please run this script from the scripts directory."
        exit 1
    fi
    
    # Check if model exists
    if [ ! -f "$PROJECT_ROOT/model/ard.40_0.00649.hdf5" ]; then
        echo "Error: Model file not found at $PROJECT_ROOT/model/ard.40_0.00649.hdf5"
        exit 1
    fi
    
    # Check if dataset directory exists
    if [ ! -d "$PROJECT_ROOT/dataset" ]; then
        echo "Error: Dataset directory not found. Please ensure images are in $PROJECT_ROOT/dataset/"
        exit 1
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/output"
    
    echo "Running ARD-CNN model locally..."
    echo "Input: $PROJECT_ROOT/dataset/"
    echo "Output: $PROJECT_ROOT/output/"
    echo "Model: $PROJECT_ROOT/model/ard.40_0.00649.hdf5"
    echo ""
    
    # Change to the removal directory and run the model
    cd "$PROJECT_ROOT/removal"
    python test_ardcnn.py
fi

echo ""
echo "Processing complete!"
echo "Check the output directory for results."