#!/bin/bash
# Script for testing and comparision of model weights

# Set source and destination directories
WEIGHTS_DIR="./weights"
DEST_DIR="."

for file in "$WEIGHTS_DIR"/*; do
    # Check if it's a file (not a directory)
    if [ -f "$file" ]; then
      echo "$file"
      cp "$file" "$DEST_DIR/best_weights.pth"
      python3 -m gupb
    fi
done