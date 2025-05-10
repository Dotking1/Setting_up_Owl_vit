import os
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

def main():
    """Simple debug script to test a single image with OWL-ViT."""
    print("\n=== OWL-ViT Debug Script ===\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get an image file from the Emergent folder
    emergent_folder = "Emergent"
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    image_files = [
        os.path.join(emergent_folder, f) 
        for f in os.listdir(emergent_folder) 
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        print(f"No image files found in '{emergent_folder}' folder.")
        return
    
    # Use the first image for debugging
    image_path = image_files[0]
    print(f"Testing with image: {image_path}")
    
    # Load image
    try:
        image = Image.open(image_path)
        print(f"Image successfully loaded. Size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Load model
    print("Loading OWL-ViT model...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model = model.to(device)
    print("Model loaded successfully")
    
    # Test with very simple text queries
    text_queries = [
        "person",
        "human",
        "a photo of a person",
        "a photo of a human",
        "man", 
        "woman"
    ]
    
    print(f"Testing detection with queries: {text_queries}")
    
    # Process image and text
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Perform inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)
    
    # Print all scores for debugging
    print("\n=== Raw Detection Results ===")
    boxes = results[0]["boxes"].cpu().numpy()
    scores = results[0]["scores"].cpu().numpy()
    labels = results[0]["labels"].tolist()
    
    if len(scores) == 0:
        print("No detections found.")
    else:
        # Print results for different thresholds
        for threshold in [0.05, 0.01, 0.001]:
            print(f"\nResults with threshold {threshold}:")
            valid_indices = [i for i, score in enumerate(scores) if score > threshold]
            
            if not valid_indices:
                print(f"  No detections above threshold {threshold}")
            else:
                for i in valid_indices:
                    box = boxes[i]
                    label_idx = labels[i]
                    label = text_queries[label_idx]
                    score = scores[i]
                    
                    print(f"  {label}: {score:.6f}, Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Print all raw scores for all classes
    print("\nAll raw scores:")
    for i, query in enumerate(text_queries):
        # Find scores for this class
        class_indices = [j for j, label_idx in enumerate(labels) if label_idx == i]
        
        if class_indices:
            class_scores = [scores[j] for j in class_indices]
            print(f"  {query}: {', '.join([f'{score:.6f}' for score in class_scores])}")
        else:
            print(f"  {query}: No detections")

if __name__ == "__main__":
    main()