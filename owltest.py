import os
import time
import torch
import requests
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transformers import OwlViTProcessor, OwlViTForObjectDetection

def draw_bounding_boxes(image, boxes, labels, scores, threshold=0.05, output_filename="output_detection.jpg"):
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
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"\nDetection results saved to '{output_filename}'")
    plt.show()

def process_image(image_path, model, processor, text_queries, device, detection_threshold=0.05, 
                output_images_dir="output_images", output_text_dir="output_text"):
    """Process a single image with OWL-ViT."""
    print(f"\nProcessing image: {image_path}")
    
    try:
        # Load image
        image = Image.open(image_path)
        print(f"Image size: {image.size}")
        
        # Process image and text
        start_time = time.time()
        inputs = processor(text=text_queries, images=image, return_tensors="pt")
        
        # Move inputs to device if using MPS or CUDA
        if device != "cpu":
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Measure inference time
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")
        
        # Post-process results
        target_sizes = torch.Tensor([image.size[::-1]])
        if device != "cpu":
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes.to(device))
        else:
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
        
        # Extract predictions
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = [text_queries[i] for i in results[0]["labels"]]
        
        # Print all scores for debugging
        print("All detection scores:")
        for i, (score, label_idx) in enumerate(zip(scores, results[0]["labels"])):
            print(f"  {text_queries[label_idx]}: {score:.4f}")
            
        # Clean up labels for display
        labels = [label.replace("a photo of a ", "").replace("a photo of ", "") for label in labels]
        
        # Generate output filenames
        base_name = os.path.basename(image_path)
        output_filename = os.path.join(output_images_dir, f"output_{os.path.splitext(base_name)[0]}.jpg")
        text_output_filename = os.path.join(output_text_dir, f"results_{os.path.splitext(base_name)[0]}.txt")
        
        # Write results to text file
        with open(text_output_filename, 'w') as f:
            f.write(f"Detection Results for: {image_path}\n")
            f.write(f"Image size: {image.size[0]}x{image.size[1]}\n")
            f.write(f"Inference time: {inference_time:.4f} seconds\n")
            f.write(f"Device: {device}\n")
            f.write(f"Detection threshold: {detection_threshold}\n\n")
            
            f.write("Detection queries:\n")
            for query in text_queries:
                f.write(f"- {query}\n")
            f.write("\n")
            
            f.write("All raw detection scores:\n")
            for i, (score, label_idx) in enumerate(zip(scores, results[0]["labels"])):
                f.write(f"- {text_queries[label_idx]}: {score:.6f}\n")
            f.write("\n")
            
            f.write(f"Objects detected (threshold = {detection_threshold}):\n")
            valid_indices = [i for i, score in enumerate(scores) if score > detection_threshold]
            
            if not valid_indices:
                f.write("No objects detected above threshold.\n")
            else:
                for i in valid_indices:
                    box = boxes[i]
                    label = labels[i]
                    score = scores[i]
                    f.write(f"- {label}: confidence={score:.6f}, coordinates=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]\n")
            
            f.write(f"\nTotal objects detected: {len(valid_indices)}\n")
            f.write(f"Output image saved as: {output_filename}\n")
        
        print(f"Detailed results saved to: {text_output_filename}")
        
        # Display results
        draw_bounding_boxes(image, boxes, labels, scores, threshold=detection_threshold, output_filename=output_filename)
        
        return True
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

def main():
    print("\n=== OWL-ViT Object Detection Demo ===\n")
    
    # Set matplotlib backend to not require a display
    # This is useful if running on a server without a display
    try:
        import matplotlib
        matplotlib.use('Agg')
    except:
        pass
    
    # Ensure output directories exist
    output_images_dir = "output_images"
    output_text_dir = "output_text"
    
    for directory in [output_images_dir, output_text_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Create a summary file for all processed images
    summary_file = os.path.join(output_text_dir, "detection_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("OWL-ViT Object Detection Summary\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # For Apple Silicon
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Define the path to the Emergent folder
    emergent_folder = "Emergent"
    
    # Check if folder exists
    if not os.path.exists(emergent_folder):
        print(f"Error: Folder '{emergent_folder}' does not exist.")
        return
    
    # Get all image files from the Emergent folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [
        os.path.join(emergent_folder, f) 
        for f in os.listdir(emergent_folder) 
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        print(f"No image files found in '{emergent_folder}' folder.")
        return
    
    print(f"Found {len(image_files)} images in '{emergent_folder}' folder.")
    
    print("\nLoading OWL-ViT model...")
    start_time = time.time()
    
    # Load OWL-ViT model and processor
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    
    # Move model to device
    if device != "cpu":
        model = model.to(device)
    
    # Measure model loading time
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Define objects to detect with better prompt formatting
    text_queries = [
        "a photo of a person",
        "a photo of a human",
        "a photo of a man",
        "a photo of a woman", 
        "a photo of a child",
        "a photo of people",
        "a photo of humans"
    ]
    
    # You can modify these classes according to your needs
    print(f"\nRunning detection for: {', '.join([q.replace('a photo of a ', '') for q in text_queries])}")
    
    # Process each image
    successful = 0
    total_time = 0
    detection_counts = {}
    
    with open(summary_file, 'a') as summary:
        for image_path in image_files:
            start_time = time.time()
            if process_image(image_path, model, processor, text_queries, device, 
                        output_images_dir=output_images_dir, output_text_dir=output_text_dir):
                successful += 1
                process_time = time.time() - start_time
                total_time += process_time
                
                # Count detections for this image
                base_name = os.path.basename(image_path)
                results_file = f"results_{os.path.splitext(base_name)[0]}.txt"
                detections_count = 0
                
                # Parse the results file to count detections
                try:
                    with open(results_file, 'r') as f:
                        for line in f:
                            if line.startswith("- ") and "confidence=" in line:
                                detections_count += 1
                                
                    detection_counts[image_path] = detections_count
                    
                    # Add to summary
                    summary.write(f"{image_path}: {detections_count} detections, processed in {process_time:.2f}s\n")
                except:
                    summary.write(f"{image_path}: Error parsing results\n")
    
    print(f"\nProcessed {successful} out of {len(image_files)} images successfully.")
    
    # Write final summary to file
    with open(summary_file, 'a') as f:
        f.write(f"\nSummary Statistics:\n")
        f.write(f"Total images processed: {successful} of {len(image_files)}\n")
        if successful > 0:
            f.write(f"Average processing time: {total_time/successful:.2f} seconds per image\n")
            f.write(f"Total objects detected: {sum(detection_counts.values())}\n")
            f.write(f"Average objects per image: {sum(detection_counts.values())/successful:.2f}\n")
        
        # List images with most detections
        if detection_counts:
            f.write("\nImages by detection count:\n")
            sorted_images = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)
            for img, count in sorted_images:
                f.write(f"- {img}: {count} detections\n")
    
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()