#!/usr/bin/env python3
"""
OWL-ViT Test Script for Mac M4
Tests the OWL-ViT model on a COCO dataset image and displays the results
"""

import os
import time
import torch
import requests
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import OwlViTProcessor, OwlViTForObjectDetection

def download_image(url, filename):
    """Download an image from a URL."""
    print(f"Downloading image from {url}...")
    
    if os.path.exists(filename):
        print(f"Image already exists: {filename}")
        return filename
        
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    print(f"Image downloaded: {filename}")
    return filename

def draw_bounding_boxes(image, boxes, labels, scores, threshold=0.1):
    """Draw bounding boxes on the image."""
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    # Filter by threshold
    valid_indices = [i for i, score in enumerate(scores) if score > threshold]
    
    print(f"\nDetection Results (threshold = {threshold}):")
    for i in valid_indices:
        box = boxes[i]
        label = labels[i]
        score = scores[i]
        
        print(f"  {label}: {score:.2f}, Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = Rectangle((x, y), width, height, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        
        plt.text(
            x, y - 5, 
            f"{label}: {score:.2f}", 
            color="white", 
            fontsize=12,
            bbox=dict(facecolor="red", alpha=0.7)
        )
    
    print(f"Total objects detected: {len(valid_indices)}")
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("output_detection.jpg", bbox_inches="tight")
    print("\nDetection results saved to 'output_detection.jpg'")
    plt.show()

def main():
    print("\n=== OWL-ViT Object Detection Demo ===\n")
    
    # Check for MPS (Metal Performance Shaders) availability
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Download a test image from COCO dataset
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = download_image(image_url, "coco_test_image.jpg")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    
    print("\nLoading OWL-ViT model...")
    start_time = time.time()
    
    # Load OWL-ViT model and processor
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    
    # Move model to device if using MPS
    if device == "mps":
        model = model.to(device)
    
    # Measure model loading time
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Define objects to detect (common COCO objects)
    text_queries = [
        "a photo of a cat",
        "a photo of a dog", 
        "a photo of a person", 
        "a photo of a car",
        "a photo of a couch",
        "a photo of a potted plant"
        "a photo of a tv remote"
    ]
    
    print(f"\nRunning detection for: {', '.join([q.replace('a photo of a ', '') for q in text_queries])}")
    
    # Process image and text
    start_time = time.time()
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    
    # Move inputs to device if using MPS
    if device == "mps":
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Measure inference time
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    # Post-process results
    target_sizes = torch.Tensor([image.size[::-1]])
    if device == "mps":
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes.to(device))
    else:
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
    
    # Extract predictions
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    labels = [text_queries[i].replace("a photo of a ", "") for i in results[0]["labels"]]
    
    # Display results
    draw_bounding_boxes(image, boxes, labels, scores, threshold=0.1)

if __name__ == "__main__":
    main()