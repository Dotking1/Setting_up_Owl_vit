import os
import requests
import json
from pathlib import Path
from tqdm import tqdm
import shutil

# Determine Hugging Face cache directory
def get_hf_cache_home():
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HOME)
    except ImportError:
        # If huggingface_hub is not installed, use default location
        return Path(os.path.expanduser("~/.cache/huggingface"))

# URL for the model repository
MODEL_REPO = "google/owlvit-base-patch32"
MODEL_REVISION = "main"  # Usually "main" or "master"

def download_file(url, dest_path):
    """Download a file with progress bar"""
    print(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Download with progress bar
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Downloaded to: {dest_path}")
    return dest_path

def main():
    # Define cache directory
    hf_home = get_hf_cache_home()
    print(f"Hugging Face cache directory: {hf_home}")
    
    # Define model directory
    model_id = MODEL_REPO.replace("/", "--")
    snapshot_dir = hf_home / "hub" / f"models--{model_id}" / "snapshots"
    
    # Create directories
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # First download the model info to get the latest commit hash
    api_url = f"https://huggingface.co/api/models/{MODEL_REPO}"
    print(f"Fetching model info from: {api_url}")
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        model_info = response.json()
        latest_commit = model_info.get("sha", "")
        
        if not latest_commit:
            print("Could not determine latest commit hash. Using a placeholder.")
            latest_commit = "placeholder-commit-hash"
    except Exception as e:
        print(f"Error fetching model info: {e}")
        print("Using placeholder commit hash.")
        latest_commit = "placeholder-commit-hash"
    
    # Create commit directory
    commit_dir = snapshot_dir / latest_commit
    os.makedirs(commit_dir, exist_ok=True)
    
    # Create refs directory and save the main ref
    refs_dir = hf_home / "hub" / f"models--{model_id}" / "refs"
    os.makedirs(refs_dir, exist_ok=True)
    
    with open(refs_dir / MODEL_REVISION, "w") as f:
        f.write(latest_commit)
    
    # Files to download
    files = [
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    ]
    
    # Download each file
    for filename in files:
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{filename}"
        dest_path = commit_dir / filename
        
        # Skip if file already exists
        if os.path.exists(dest_path):
            print(f"File already exists: {dest_path}")
            continue
        
        try:
            download_file(url, dest_path)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"File not found: {filename}, skipping...")
            else:
                print(f"Error downloading {filename}: {e}")
    
    print("\nDownload complete! The model should now be cached locally.")
    print(f"Model directory: {commit_dir}")

if __name__ == "__main__":
    main()